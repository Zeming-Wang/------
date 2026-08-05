<script setup>
import { computed, ref, watch } from 'vue';
import MapEditor from './MapEditor.vue';
import StringListEditor from './StringListEditor.vue';
import ToolListEditor from './ToolListEditor.vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  selectedNode: { type: Object, default: null },
  document: { type: Object, required: true },
  keyPool: { type: Object, required: true }
});

const emit = defineEmits(['update-node', 'delete-node', 'open-loop']);

const { isChinese, t } = useI18n();
const activeNodeTab = ref('basics');

const agentTabs = computed(() => [
  { id: 'basics', label: isChinese.value ? '基础' : 'Basics' },
  { id: 'keys', label: isChinese.value ? '字段' : 'Keys' },
  { id: 'rules', label: isChinese.value ? '规则' : 'Rules' },
  { id: 'tools', label: t('tools') }
]);

const customTabs = computed(() => [
  { id: 'basics', label: isChinese.value ? '基础' : 'Basics' },
  { id: 'templates', label: t('templates') },
  { id: 'static', label: t('staticOutputs') },
  { id: 'pick', label: t('pickKeys') },
  { id: 'keys', label: isChinese.value ? '字段' : 'Keys' }
]);

const activeTabs = computed(() => {
  if (props.selectedNode?.type === 'custom') return customTabs.value;
  return agentTabs.value;
});
const activeTabOptions = computed(() => activeTabs.value.map((tab) => ({ label: tab.label, value: tab.id })));

watch(
  () => props.selectedNode?.id || '',
  () => {
    activeNodeTab.value = 'basics';
  }
);

watch(
  () => props.selectedNode?.type || '',
  () => {
    const validTab = activeTabs.value.some((tab) => tab.id === activeNodeTab.value);
    if (!validTab) activeNodeTab.value = 'basics';
  }
);

const nodeBehaviorRuleSuggestions = computed(() =>
  isChinese.value
    ? ['保持答案简洁。', '只使用提供的上下文。', '简要列出假设。', '返回结构化输出。']
    : ['Keep the answer concise.', 'Only use the provided context.', 'List assumptions briefly.', 'Return structured output.']
);
const customModeOptions = computed(() => [
  { value: 'passthrough', label: isChinese.value ? '透传' : 'Passthrough' },
  { value: 'template', label: t('templates') },
  { value: 'set', label: isChinese.value ? '设置静态值' : 'Set Static Values' },
  { value: 'pick', label: t('pickKeys') },
  { value: 'compose', label: isChinese.value ? '组合混合输出' : 'Compose Mixed Output' }
]);

function updateNodeField(field, value) {
  if (!props.selectedNode) return;
  emit('update-node', { id: props.selectedNode.id, patch: { [field]: value } });
}

function updateNodeConfig(field, value) {
  if (!props.selectedNode) return;
  emit('update-node', {
    id: props.selectedNode.id,
    patch: { config: { ...props.selectedNode.config, [field]: value } }
  });
}

function mergeSuggestions(...groups) {
  const map = new Map();
  for (const group of groups) {
    for (const item of group || []) {
      const key = String(item?.key || '').trim();
      if (!key) continue;
      if (!map.has(key)) {
        map.set(key, { key, value: String(item?.value || '') });
      }
    }
  }
  return [...map.values()];
}

function mappingSuggestions(mapping) {
  return Object.entries(mapping || {}).map(([key, value]) => ({
    key: String(key),
    value: String(value ?? '')
  }));
}

function incomingEdges(nodeId) {
  return (props.document?.edges || []).filter((edge) => edge.target === nodeId);
}

function outgoingEdges(nodeId) {
  return (props.document?.edges || []).filter((edge) => edge.source === nodeId);
}

const globalKeySuggestions = computed(() => mappingSuggestions(props.keyPool?.key_map || {}));
const selectedIncomingSuggestions = computed(() => {
  if (!props.selectedNode) return [];
  return mergeSuggestions(...incomingEdges(props.selectedNode.id).map((edge) => mappingSuggestions(edge.mapping)));
});
const selectedOutgoingSuggestions = computed(() => {
  if (!props.selectedNode) return [];
  return mergeSuggestions(...outgoingEdges(props.selectedNode.id).map((edge) => mappingSuggestions(edge.mapping)));
});
const incomingKeyNames = computed(() => selectedIncomingSuggestions.value.map((item) => item.key));
const outgoingKeyNames = computed(() => selectedOutgoingSuggestions.value.map((item) => item.key));

const agentPullSuggestions = computed(() =>
  mergeSuggestions(globalKeySuggestions.value, selectedIncomingSuggestions.value, [
    { key: 'message', value: 'Input message' },
    { key: 'context', value: 'Context' }
  ])
);
const agentPushSuggestions = computed(() =>
  mergeSuggestions(globalKeySuggestions.value, selectedOutgoingSuggestions.value, [
    { key: 'answer', value: 'Main answer' },
    { key: 'summary', value: 'Summary' }
  ])
);
const customPullSuggestions = computed(() =>
  mergeSuggestions(globalKeySuggestions.value, selectedIncomingSuggestions.value, [
    { key: 'message', value: 'Input message' },
    { key: 'context', value: 'Context' }
  ])
);
const customPushSuggestions = computed(() =>
  mergeSuggestions(globalKeySuggestions.value, selectedOutgoingSuggestions.value, [
    { key: 'message', value: 'Output message' },
    { key: 'result', value: 'Result' }
  ])
);

function extractPlaceholders(text) {
  const matches = String(text || '').match(/\{([^{}]+)\}/g) || [];
  return [...new Set(matches.map((item) => item.slice(1, -1).trim()).filter(Boolean))];
}

const promptWarnings = computed(() => {
  if (props.selectedNode?.type !== 'agent') return [];
  const allowed = new Set(Object.keys(props.selectedNode.config?.pull_keys || {}));
  const global = new Set(props.keyPool?.key_names || []);
  return extractPlaceholders(props.selectedNode.config?.prompt_template || '').map((key) => {
    if (allowed.has(key)) return null;
    if (global.has(key)) {
      return isChinese.value
        ? `提示模板：{${key}} 在可用字段中，但未加入 pull_keys。`
        : `Prompt template: {${key}} is in available fields but missing from pull_keys.`;
    }
    return isChinese.value
      ? `提示模板：无法在 pull_keys 或可用字段中找到 {${key}}。`
      : `Prompt template: {${key}} cannot be found in pull_keys or available fields.`;
  }).filter(Boolean);
});

function mappingValueWarnings(mapping, allowedKeys, globalKeys, scopeLabel) {
  const allowed = new Set(Object.keys(allowedKeys || {}));
  const global = new Set(globalKeys || []);
  const warnings = [];
  for (const [field, value] of Object.entries(mapping || {})) {
    for (const key of extractPlaceholders(String(value || ''))) {
      if (allowed.has(key)) continue;
      if (global.has(key)) {
        warnings.push(isChinese.value
          ? `${scopeLabel} '${field}'：{${key}} 在可用字段中，但未加入 pull_keys。`
          : `${scopeLabel} '${field}': {${key}} is in available fields but missing from pull_keys.`);
      } else {
        warnings.push(isChinese.value
          ? `${scopeLabel} '${field}'：无法在 pull_keys 或可用字段中找到 {${key}}。`
          : `${scopeLabel} '${field}': {${key}} cannot be found in pull_keys or available fields.`);
      }
    }
  }
  return warnings;
}

const customTemplateWarnings = computed(() =>
  mappingValueWarnings(
    props.selectedNode?.config?.templates || {},
    props.selectedNode?.config?.pull_keys || {},
    props.keyPool?.key_names || [],
    isChinese.value ? '逻辑节点模板' : 'Logic node templates'
  )
);

const customStaticWarnings = computed(() =>
  mappingValueWarnings(
    props.selectedNode?.config?.static_outputs || {},
    props.selectedNode?.config?.pull_keys || {},
    props.keyPool?.key_names || [],
    isChinese.value ? '逻辑节点静态输出' : 'Logic node static outputs'
  )
);
</script>

<template>
  <section class="panel-section loop-panel-scroll">
    <div class="section-title">{{ t('selectedInnerNode') }}</div>
    <template v-if="selectedNode">
      <el-segmented
        v-if="selectedNode.type === 'agent' || selectedNode.type === 'custom'"
        v-model="activeNodeTab"
        class="inline-tabs compact-tabs"
        :options="activeTabOptions"
      />

      <template v-if="selectedNode.type === 'agent'">
        <template v-if="activeNodeTab === 'basics'">
          <label>
            <span>{{ t('label') }}</span>
            <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
          </label>
          <label>
            <span>{{ t('type') }}</span>
            <el-input :model-value="selectedNode.type" disabled />
          </label>
          <label>
            <span>{{ t('instructions') }}</span>
            <el-input
              :model-value="selectedNode.config.instructions || ''"
              type="textarea"
              :autosize="{ minRows: 3, maxRows: 8 }"
              @update:model-value="updateNodeConfig('instructions', $event)"
            />
          </label>
          <label>
            <span>{{ t('promptTemplate') }}</span>
            <el-input
              :model-value="selectedNode.config.prompt_template || ''"
              type="textarea"
              :autosize="{ minRows: 3, maxRows: 10 }"
              @update:model-value="updateNodeConfig('prompt_template', $event)"
            />
          </label>
          <div v-if="promptWarnings.length" class="warning-block">
            <div class="warning-title">{{ isChinese ? '提示词警告' : 'Prompt Warnings' }}</div>
            <div v-for="warning in promptWarnings" :key="warning" class="field-help warning-text">{{ warning }}</div>
          </div>
        </template>

        <template v-else-if="activeNodeTab === 'keys'">
          <div class="subsection-title">{{ t('pullKeys') }}</div>
          <MapEditor
            :value="selectedNode.config.pull_keys || {}"
            :key-label="t('inputKey')"
            :value-label="t('meaning')"
            :suggestions="agentPullSuggestions"
            :suggestion-title="t('workflowKeyPool')"
            key-input-mode="select"
            :show-suggestion-strip="false"
            @update:value="updateNodeConfig('pull_keys', $event)"
          />
          <div v-if="incomingKeyNames.length" class="field-help">{{ isChinese ? '上游可达字段' : 'Upstream reachable keys' }}: <code>{{ incomingKeyNames.join(', ') }}</code></div>

          <div class="subsection-title">{{ t('pushKeys') }}</div>
          <MapEditor
            :value="selectedNode.config.push_keys || {}"
            :key-label="t('outputKey')"
            :value-label="t('meaning')"
            :suggestions="agentPushSuggestions"
            :suggestion-title="t('workflowKeyPool')"
            key-input-mode="select"
            :show-suggestion-strip="false"
            @update:value="updateNodeConfig('push_keys', $event)"
          />
          <div v-if="outgoingKeyNames.length" class="field-help">{{ isChinese ? '下游连线引用' : 'Downstream edges reference' }}: <code>{{ outgoingKeyNames.join(', ') }}</code></div>
        </template>

        <StringListEditor
          v-else-if="activeNodeTab === 'rules'"
          :value="selectedNode.config.behavior_rules || []"
          :placeholder="isChinese ? '保持答案简洁。' : 'Keep the answer concise.'"
          :suggestions="nodeBehaviorRuleSuggestions"
          :suggestion-title="t('commonNodeRules')"
          @update:value="updateNodeConfig('behavior_rules', $event)"
        />

        <ToolListEditor
          v-else
          :value="selectedNode.config.tools || []"
          :title="isChinese ? '这一阶段可以使用的工具' : 'Tools This Stage Can Use'"
          :help="isChinese ? '这些工具只作用于该 Loop 内部推理阶段。' : 'These tools only apply to this inner reasoning stage.'"
          @update:value="updateNodeConfig('tools', $event)"
        />
      </template>

      <template v-else-if="selectedNode.type === 'custom'">
        <template v-if="activeNodeTab === 'basics'">
          <label>
            <span>{{ t('label') }}</span>
            <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
          </label>
          <label>
            <span>{{ t('type') }}</span>
            <el-input :model-value="selectedNode.type" disabled />
          </label>
          <label>
            <span>{{ t('mode') }}</span>
            <el-select :model-value="selectedNode.config.mode || 'passthrough'" @update:model-value="updateNodeConfig('mode', $event)">
              <el-option v-for="option in customModeOptions" :key="option.value" :value="option.value" :label="option.label" />
            </el-select>
          </label>
          <div class="field-help">
            {{ isChinese ? 'compose 允许内部逻辑节点在一步中组合 pick、template 和 static outputs。' : 'compose lets an inner logic node combine pick, template, and static outputs in one step.' }}
          </div>
        </template>

        <template v-else-if="activeNodeTab === 'templates'">
          <MapEditor
            :value="selectedNode.config.templates || {}"
            :key-label="t('outputKey')"
            :value-label="isChinese ? '模板' : 'Template'"
            :suggestions="customPushSuggestions"
            :suggestion-title="t('workflowKeyPool')"
            key-input-mode="select"
            :show-suggestion-strip="false"
            @update:value="updateNodeConfig('templates', $event)"
          />
          <div v-if="customTemplateWarnings.length" class="warning-block">
            <div class="warning-title">{{ isChinese ? '模板警告' : 'Template Warnings' }}</div>
            <div v-for="warning in customTemplateWarnings" :key="warning" class="field-help warning-text">{{ warning }}</div>
          </div>
        </template>

        <template v-else-if="activeNodeTab === 'static'">
          <MapEditor
            :value="selectedNode.config.static_outputs || {}"
            :key-label="t('outputKey')"
            :value-label="isChinese ? '静态值' : 'Static Value'"
            :suggestions="customPushSuggestions"
            :suggestion-title="t('workflowKeyPool')"
            key-input-mode="select"
            :show-suggestion-strip="false"
            @update:value="updateNodeConfig('static_outputs', $event)"
          />
          <div v-if="customStaticWarnings.length" class="warning-block">
            <div class="warning-title">{{ isChinese ? '静态输出警告' : 'Static Output Warnings' }}</div>
            <div v-for="warning in customStaticWarnings" :key="warning" class="field-help warning-text">{{ warning }}</div>
          </div>
        </template>

        <MapEditor
          v-else-if="activeNodeTab === 'pick'"
          :value="selectedNode.config.pick_keys || {}"
          :key-label="t('outputKey')"
          :value-label="isChinese ? '来源字段' : 'Source Key'"
          :suggestions="customPullSuggestions"
          :suggestion-title="t('workflowKeyPool')"
          suggestion-value-mode="key"
          key-input-mode="select"
          :show-suggestion-strip="false"
          @update:value="updateNodeConfig('pick_keys', $event)"
        />

        <template v-else>
          <div class="subsection-title">{{ t('pullKeys') }}</div>
          <MapEditor
            :value="selectedNode.config.pull_keys || {}"
            :key-label="t('inputKey')"
            :value-label="t('meaning')"
            :suggestions="customPullSuggestions"
            :suggestion-title="t('workflowKeyPool')"
            key-input-mode="select"
            :show-suggestion-strip="false"
            @update:value="updateNodeConfig('pull_keys', $event)"
          />
          <div v-if="incomingKeyNames.length" class="field-help">{{ isChinese ? '上游可达字段' : 'Upstream reachable keys' }}: <code>{{ incomingKeyNames.join(', ') }}</code></div>

          <div class="subsection-title">{{ t('pushKeys') }}</div>
          <MapEditor
            :value="selectedNode.config.push_keys || {}"
            :key-label="t('outputKey')"
            :value-label="t('meaning')"
            :suggestions="customPushSuggestions"
            :suggestion-title="t('workflowKeyPool')"
            key-input-mode="select"
            :show-suggestion-strip="false"
            @update:value="updateNodeConfig('push_keys', $event)"
          />
          <div v-if="selectedNode.config.mode === 'compose'" class="field-help">{{ isChinese ? 'Compose 模式会按 pick、template、static outputs 的顺序应用输出。' : 'Compose mode applies outputs in the order pick, template, then static outputs.' }}</div>
          <div v-if="outgoingKeyNames.length" class="field-help">{{ isChinese ? '下游连线引用' : 'Downstream edges reference' }}: <code>{{ outgoingKeyNames.join(', ') }}</code></div>
        </template>
      </template>

      <template v-else-if="selectedNode.type === 'loop'">
        <label>
          <span>{{ t('label') }}</span>
          <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
        </label>
        <label>
          <span>{{ t('type') }}</span>
          <el-input :model-value="selectedNode.type" disabled />
        </label>
        <div class="field-help">
          {{ isChinese ? '这是嵌套 Loop 节点。打开它自己的编辑器来设计内部节点、控制器输入映射和控制器输出映射。' : 'This is a nested loop node. Open its own editor to design inner nodes, controller input mappings, and controller output mappings.' }}
        </div>
        <el-button type="primary" @click="$emit('open-loop', selectedNode.id)">{{ t('editNestedLoop') }}</el-button>
      </template>

      <el-button type="danger" @click="$emit('delete-node', selectedNode.id)">{{ t('deleteNode') }}</el-button>
    </template>
    <p v-else class="empty-state">{{ t('selectInnerNodeEmpty') }}</p>
  </section>
</template>
