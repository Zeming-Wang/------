<script setup>
import { computed, ref, watch } from 'vue';
import BehaviorEditor from './BehaviorEditor.vue';
import MapEditor from './MapEditor.vue';
import StringListEditor from './StringListEditor.vue';
import ToolListEditor from './ToolListEditor.vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  selectedNode: { type: Object, default: null },
  selectedEdge: { type: Object, default: null },
  edgeSuggestions: { type: Array, default: () => [] },
  manifest: { type: Object, required: true },
  document: { type: Object, required: true },
  keyPool: { type: Object, required: true }
});

const emit = defineEmits([
  'update-node',
  'update-manifest',
  'update-edge-mapping',
  'ai-author-field',
  'delete-node',
  'delete-edge',
  'open-loop-editor',
  'open-custom-editor'
]);

const detailGroup = ref('');
const detailSection = ref('');
const nodeModalGroups = new Set(['node', 'agent', 'custom', 'loop']);
const { isChinese, t } = useI18n();

const behaviorRuleSuggestions = [
  'Explain the basis of the answer.',
  'Do not invent missing facts.',
  'Keep the output actionable.',
  'Separate assumptions from conclusions.'
];
const terminateModeOptions = computed(() => [
  { value: 'never', label: t('neverStopEarly') },
  { value: 'key_truthy', label: t('stopWhenKeyTruthy') },
  { value: 'key_equals', label: t('stopWhenKeyEqualsValue') }
]);
const keyRuleHelp = computed(() =>
  isChinese.value
    ? '字段名应使用 query、analysis_result、answer、done 这类标识符。建议使用小写字母、数字和下划线。节点可以定义新字段；连线负责把上游输出字段映射到下游输入字段。'
    : 'Key names are identifiers like query, analysis_result, answer, done. Use lowercase letters, numbers, and underscores when possible. Nodes may define new keys; edges map one node output key to another node input key.'
);

const manifestNav = computed(() => [
  { id: 'overview', label: t('overview') },
  { id: 'derived', label: t('workflowDerived') },
  { id: 'behavior', label: t('globalRules') }
]);

const agentNav = computed(() => [
  { id: 'basics', label: t('basics') },
  { id: 'prompt', label: t('prompt') },
  { id: 'keys', label: t('keys') },
  { id: 'tools', label: t('tools') }
]);

const customNav = computed(() => [
  { id: 'basics', label: t('basics') },
  { id: 'runtime', label: t('runtimeView') }
]);

const loopNav = computed(() => [
  { id: 'basics', label: t('basics') },
  { id: 'termination', label: t('termination') },
  { id: 'structure', label: t('structure') }
]);

const genericNodeNav = computed(() => [
  { id: 'basics', label: t('basics') }
]);

watch(
  () => props.selectedNode?.id || '',
  () => {
    if (nodeModalGroups.has(detailGroup.value)) {
      closeDetail();
    }
  }
);

function defaultSectionForGroup(group) {
  if (group === 'manifest') return 'overview';
  if (group === 'agent') return 'basics';
  if (group === 'custom') return 'basics';
  if (group === 'loop') return 'basics';
  if (group === 'node') return 'basics';
  return '';
}

function openDetail(group, section = '') {
  detailGroup.value = group;
  detailSection.value = section || defaultSectionForGroup(group);
}

function closeDetail() {
  detailGroup.value = '';
  detailSection.value = '';
}

function updateNodeField(field, value) {
  if (!props.selectedNode) return;
  emit('update-node', {
    id: props.selectedNode.id,
    patch: { [field]: value }
  });
}

function updateNodeConfig(field, value) {
  if (!props.selectedNode) return;
  emit('update-node', {
    id: props.selectedNode.id,
    patch: {
      config: {
        ...props.selectedNode.config,
        [field]: value
      }
    }
  });
}

function updateNestedConfig(rootField, field, value) {
  if (!props.selectedNode) return;
  const root = props.selectedNode.config[rootField] || {};
  emit('update-node', {
    id: props.selectedNode.id,
    patch: {
      config: {
        ...props.selectedNode.config,
        [rootField]: {
          ...root,
          [field]: value
        }
      }
    }
  });
}

function updateManifest(field, value) {
  emit('update-manifest', { [field]: value });
}

function aiPolishManifestDescription() {
  emit(
    'ai-author-field',
    'manifest.description',
    props.manifest.description || '',
    (value) => updateManifest('description', value),
    'polish'
  );
}

function aiPolishManifestBehavior() {
  emit(
    'ai-author-field',
    'behavior_rules',
    (props.manifest.behavior?.rules || []).join('\n'),
    (value) => updateManifest('behavior', {
      ...(props.manifest.behavior || {}),
      rules: splitAiLines(value)
    }),
    'polish'
  );
}

function aiPolishNodeConfig(field) {
  if (!props.selectedNode) return;
  const currentValue = props.selectedNode.config?.[field] || '';
  emit(
    'ai-author-field',
    `agent.${field}`,
    currentValue,
    (value) => updateNodeConfig(field, value),
    'polish'
  );
}

function splitAiLines(value) {
  return String(value || '')
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*]\s*/, '').replace(/^\d+[.)]\s*/, '').trim())
    .filter(Boolean);
}

function updateSelectedEdgeMapping(mapping) {
  if (!props.selectedEdge) return;
  emit('update-edge-mapping', { edgeId: props.selectedEdge.id, mapping });
}

function mergeSuggestions(...groups) {
  const map = new Map();
  for (const group of groups) {
    for (const item of group || []) {
      const key = String(item?.key || '').trim();
      if (!key) continue;
      if (!map.has(key)) {
        map.set(key, { key, value: String(item?.value || '').trim() });
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

function visitWorkflowNodes(visitor) {
  function visit(node, path = []) {
    if (!node) return;
    visitor(node, path);
    for (const inner of node.config?.subgraph?.nodes || []) {
      visit(inner, [...path, node]);
    }
  }
  for (const node of props.document?.nodes || []) {
    visit(node, []);
  }
}

function slugWords(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, ' ')
    .split(/\s+/)
    .map((item) => item.trim())
    .filter((item) => item.length >= 3)
    .slice(0, 4);
}

const derivedTags = computed(() => {
  const tags = new Set(['workflow']);
  const typeCounts = {};
  visitWorkflowNodes((node) => {
    typeCounts[node.type] = (typeCounts[node.type] || 0) + 1;
    if (node.type === 'agent') tags.add('reasoning');
    if (node.type === 'custom') tags.add('logic');
    if (node.type === 'loop') tags.add('loop');
  });
  if ((typeCounts.agent || 0) > 1) tags.add('multi-reasoning');
  for (const word of slugWords(props.manifest?.name || props.document?.name)) tags.add(word);
  for (const word of slugWords(props.manifest?.description || props.document?.description)) tags.add(word);
  return [...tags].slice(0, 8);
});

const derivedTools = computed(() => {
  const tools = [];
  const seen = new Set();
  function addTool(tool, source) {
    const name = String(tool?.name || '').trim();
    const binding = String(tool?.binding || 'builtin').trim() || 'builtin';
    if (!name) return;
    const key = `${binding}:${name}`;
    if (seen.has(key)) return;
    seen.add(key);
    tools.push({
      name,
      binding,
      source,
      description: String(tool?.description || '').trim()
    });
  }

  visitWorkflowNodes((node) => {
    const label = node.label || node.id;
    for (const tool of node.config?.tools || []) {
      addTool(tool, label);
    }
    if (node.type === 'custom' && String(node.config?.mode || '').toLowerCase() === 'python') {
      addTool(
        {
          name: `${node.id}_script`,
          binding: 'script',
          description: isChinese.value
            ? `执行逻辑节点 ${label} 的确定性逻辑。`
            : `Run deterministic logic for ${label}.`
        },
        label
      );
    }
  });
  return tools;
});

const globalKeySuggestions = computed(() => mappingSuggestions(props.keyPool?.key_map || {}));

function incomingEdges(nodeId) {
  return (props.document?.edges || []).filter((edge) => edge.target === nodeId);
}

function outgoingEdges(nodeId) {
  return (props.document?.edges || []).filter((edge) => edge.source === nodeId);
}

const selectedIncomingSuggestions = computed(() => {
  if (!props.selectedNode) return [];
  return mergeSuggestions(...incomingEdges(props.selectedNode.id).map((edge) => mappingSuggestions(edge.mapping)));
});

const selectedOutgoingSuggestions = computed(() => {
  if (!props.selectedNode) return [];
  return mergeSuggestions(...outgoingEdges(props.selectedNode.id).map((edge) => mappingSuggestions(edge.mapping)));
});

const agentPullSuggestions = computed(() =>
  mergeSuggestions(globalKeySuggestions.value, selectedIncomingSuggestions.value, [
    { key: 'query', value: 'Original user request' },
    { key: 'message', value: 'Input message' },
    { key: 'context', value: 'Relevant context' }
  ])
);

const agentPushSuggestions = computed(() =>
  mergeSuggestions(globalKeySuggestions.value, selectedOutgoingSuggestions.value, [
    { key: 'answer', value: 'Main answer' },
    { key: 'analysis', value: 'Structured analysis' },
    { key: 'summary', value: 'Short summary' }
  ])
);

const incomingKeyNames = computed(() => selectedIncomingSuggestions.value.map((item) => item.key));
const outgoingKeyNames = computed(() => selectedOutgoingSuggestions.value.map((item) => item.key));

function extractPlaceholders(text) {
  const matches = String(text || '').match(/\{([^{}]+)\}/g) || [];
  return [...new Set(matches.map((item) => item.slice(1, -1).trim()).filter(Boolean))];
}

function promptWarnings(promptTemplate, allowedKeys, globalKeys, scopeLabel) {
  const allowed = new Set(Object.keys(allowedKeys || {}));
  const global = new Set((globalKeys || []).map((item) => String(item)));
  return extractPlaceholders(promptTemplate)
    .map((key) => {
      if (allowed.has(key)) return null;
      if (global.has(key)) {
        return `${scopeLabel}: {${key}} is in available fields but missing from pull_keys.`;
      }
      return `${scopeLabel}: {${key}} cannot be found in pull_keys or available fields.`;
    })
    .filter(Boolean);
}

const agentPromptWarnings = computed(() =>
  promptWarnings(
    props.selectedNode?.config?.prompt_template || '',
    props.selectedNode?.config?.pull_keys || {},
    props.keyPool?.key_names || [],
    'Prompt template'
  )
);

const nodeInputMap = computed(() => {
  const node = props.selectedNode;
  if (!node) return {};
  if (node.type === 'start') return props.document?.inputs || {};
  if (node.type === 'agent' || node.type === 'custom') return node.config?.pull_keys || {};
  if (node.type === 'loop') {
    const merged = {};
    for (const item of node.config?.controller_inputs || []) {
      Object.assign(merged, item.mapping || {});
    }
    return merged;
  }
  return {};
});

const nodeOutputMap = computed(() => {
  const node = props.selectedNode;
  if (!node) return {};
  if (node.type === 'start') return props.document?.inputs || {};
  if (node.type === 'agent' || node.type === 'custom') return node.config?.push_keys || {};
  if (node.type === 'loop') {
    const merged = {};
    for (const item of node.config?.controller_outputs || []) {
      Object.assign(merged, item.mapping || {});
    }
    return merged;
  }
  return {};
});

const modalOpen = computed(() => Boolean(detailGroup.value));
const selectedNodeTypeLabel = computed(() => {
  if (!props.selectedNode) return '';
  return isChinese.value ? `${props.selectedNode.type} 节点` : `${props.selectedNode.type} node`;
});

const selectedEdgeTitle = computed(() => {
  if (!props.selectedEdge) return '';
  return `${props.selectedEdge.source} -> ${props.selectedEdge.target}`;
});

const canDeleteSelectedNode = computed(() =>
  Boolean(props.selectedNode && props.selectedNode.type !== 'start' && props.selectedNode.type !== 'end')
);

const currentNavSections = computed(() => {
  if (detailGroup.value === 'manifest') return manifestNav.value;
  if (detailGroup.value === 'agent') return agentNav.value;
  if (detailGroup.value === 'custom') return customNav.value;
  if (detailGroup.value === 'loop') return loopNav.value;
  if (detailGroup.value === 'node') return genericNodeNav.value;
  return [];
});

const activeModalTab = computed({
  get: () => detailSection.value || defaultSectionForGroup(detailGroup.value),
  set: (value) => {
    detailSection.value = value;
  }
});

const modalTitle = computed(() => {
  if (detailGroup.value === 'manifest') return isChinese.value ? 'Skill 详细设置' : 'Skill Settings';
  if (!props.selectedNode) return t('rightInspector');
  if (detailGroup.value === 'agent') {
    return `${props.selectedNode.label} · ${isChinese.value ? '推理节点编辑器' : 'Reasoning Node Editor'}`;
  }
  if (detailGroup.value === 'custom') {
    return `${props.selectedNode.label} · ${isChinese.value ? '逻辑节点' : 'Logic Node'}`;
  }
  if (detailGroup.value === 'loop') return `${props.selectedNode.label} · ${t('loopSettings')}`;
  if (detailGroup.value === 'node') return `${props.selectedNode.label} · ${t('nodeBasics')}`;
  return t('rightInspector');
});

const modalDescription = computed(() => {
  if (detailGroup.value === 'manifest') {
    return isChinese.value
      ? '在这里编辑标准 Skill 元数据。标签和工具从当前工作流派生，全局规则可手动维护。'
      : 'Edit standard Skill metadata here. Tags and tools are derived from the workflow; global rules remain editable.';
  }
  if (detailGroup.value === 'agent') {
    return isChinese.value
      ? '在这里配置选中的推理节点，分区编辑提示词、工作流字段和这一阶段可以使用的工具。'
      : 'Configure the selected reasoning node here. Edit its prompt, workflow keys, and stage tools section by section.';
  }
  if (detailGroup.value === 'custom') {
    return isChinese.value
      ? '在这里查看选中的逻辑节点，然后打开逻辑节点编辑器编写或更新逻辑代码。'
      : 'Review the selected logic node here, then open the logic node editor to write or update its logic code.';
  }
  if (detailGroup.value === 'loop') {
    return isChinese.value
      ? '在这里编辑 Loop 设置，然后打开 Loop 编辑器构建内部子图和控制器连接。'
      : 'Edit loop settings here, then open the loop editor to build the inner subgraph and controller connections.';
  }
  if (detailGroup.value === 'node') {
    return isChinese.value
      ? '在这里更新选中节点标签并查看基础配置。'
      : 'Update the selected node label and review its basic configuration here.';
  }
  return '';
});
</script>

<template>
  <div class="inspector-panel">
    <section class="panel-section inspector-summary-card">
      <div class="section-title">{{ t('skillSettings') }}</div>
      <div class="helper-text">
        {{ t('manifestSummaryHelp') }}
      </div>

      <label>
        <span>{{ t('name') }}</span>
        <el-input :model-value="manifest.name" @update:model-value="updateManifest('name', $event)" />
      </label>
      <label>
        <span>{{ t('version') }}</span>
        <el-input :model-value="manifest.version" @update:model-value="updateManifest('version', $event)" />
      </label>
      <label>
        <span class="label-action-row">
          <span>{{ t('description') }}</span>
          <span class="inline-actions">
            <el-button size="small" @click="aiPolishManifestDescription">{{ t('aiPolish') }}</el-button>
          </span>
        </span>
        <el-input
          :model-value="manifest.description"
          type="textarea"
          :autosize="{ minRows: 3, maxRows: 6 }"
          @update:model-value="updateManifest('description', $event)"
        />
      </label>

      <el-button @click="openDetail('manifest', 'overview')">{{ t('skillDetailedSettings') }}</el-button>
    </section>

    <section class="panel-section inspector-summary-card">
      <div class="section-title">{{ selectedEdge ? t('selectedEdge') : t('selectedNode') }}</div>
      <template v-if="selectedNode">
        <div class="field-help"><code>{{ selectedNode.id }}</code> · {{ selectedNodeTypeLabel }}</div>
        <label>
          <span>{{ t('label') }}</span>
          <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
        </label>

        <div class="subsection-title">{{ t('inputFields') }}</div>
        <div class="field-pill-list">
          <span v-for="(_, key) in nodeInputMap" :key="`in-${key}`" class="field-pill">{{ key }}</span>
          <span v-if="!Object.keys(nodeInputMap).length" class="field-empty">{{ isChinese ? '无' : 'None' }}</span>
        </div>

        <div class="subsection-title">{{ t('outputFields') }}</div>
        <div class="field-pill-list">
          <span v-for="(_, key) in nodeOutputMap" :key="`out-${key}`" class="field-pill">{{ key }}</span>
          <span v-if="!Object.keys(nodeOutputMap).length" class="field-empty">{{ isChinese ? '无' : 'None' }}</span>
        </div>

        <label v-if="selectedNode.type === 'agent'">
          <span>{{ t('promptTemplate') }}</span>
          <el-input
            :model-value="selectedNode.config.prompt_template || ''"
            type="textarea"
            :autosize="{ minRows: 2, maxRows: 5 }"
            @update:model-value="updateNodeConfig('prompt_template', $event)"
          />
        </label>

        <div class="inspector-action-list">
          <template v-if="selectedNode.type === 'agent'">
            <el-button type="primary" @click="openDetail('agent', 'prompt')">{{ t('openAgentEditor') }}</el-button>
          </template>

          <template v-else-if="selectedNode.type === 'custom'">
            <el-button type="primary" @click="$emit('open-custom-editor', selectedNode.id)">{{ t('editCustomNode') }}</el-button>
          </template>

          <template v-else-if="selectedNode.type === 'loop'">
            <el-button type="primary" @click="$emit('open-loop-editor', selectedNode.id)">{{ t('editLoop') }}</el-button>
          </template>

          <template v-else>
            <el-button @click="openDetail('node', 'basics')">{{ t('openNodeBasics') }}</el-button>
          </template>

          <el-button
            v-if="canDeleteSelectedNode"
            type="danger"
            @click="$emit('delete-node', selectedNode.id)"
          >
            {{ t('deleteNode') }}
          </el-button>
        </div>
      </template>

      <template v-else-if="selectedEdge">
        <div>
          <div class="field-help"><code>{{ selectedEdge.id }}</code></div>
          <h3 class="inspector-node-title">{{ selectedEdgeTitle }}</h3>
        </div>

        <div class="inspector-kv edge-endpoints">
          <div class="inspector-kv-row">
            <span>{{ t('sourceNode') }}</span>
            <strong>{{ selectedEdge.source }}</strong>
          </div>
          <div class="inspector-kv-row">
            <span>{{ t('targetNode') }}</span>
            <strong>{{ selectedEdge.target }}</strong>
          </div>
        </div>

        <MapEditor
          :value="selectedEdge.mapping || {}"
          :key-label="t('edgeKey')"
          :value-label="t('meaning')"
          key-placeholder="message"
          :value-placeholder="t('fieldCarriesPlaceholder')"
          :help="t('edgeMappingHelp')"
          :suggestions="edgeSuggestions"
          :suggestion-title="t('fromConnectedNodes')"
          key-input-mode="select"
          :show-suggestion-strip="false"
          @update:value="updateSelectedEdgeMapping"
        />

        <div class="inspector-action-list">
          <el-button type="danger" @click="$emit('delete-edge', selectedEdge.id)">
            {{ t('deleteEdge') }}
          </el-button>
        </div>
      </template>

      <p v-else class="empty-state">
        {{ t('selectElementEmpty') }}
      </p>
    </section>

    <Teleport to="body">
      <div v-if="modalOpen" class="loop-modal-backdrop" @click.self="closeDetail">
        <div class="loop-modal-shell inspector-modal-shell">
          <header class="loop-modal-header">
            <div>
              <div class="eyebrow">{{ t('rightInspector') }}</div>
              <h2>{{ modalTitle }}</h2>
              <p>{{ modalDescription }}</p>
            </div>
            <el-button @click="closeDetail">{{ t('close') }}</el-button>
          </header>

          <main class="inspector-modal-main inspector-modal-tabs">
            <el-tabs v-model="activeModalTab" class="settings-tabs">
              <el-tab-pane
                v-for="item in currentNavSections"
                :key="item.id"
                :name="item.id"
                :label="item.label"
              />
            </el-tabs>

              <section v-if="detailGroup === 'manifest' && detailSection === 'overview'" class="panel-section">
                <div class="section-title">{{ t('skillMetadata') }}</div>
                <div class="helper-text">
                  {{ isChinese ? '这些字段会写入标准 Skill 的 frontmatter。描述应说明 Skill 做什么、什么时候用。' : 'These fields are written into standard Skill frontmatter. The description should explain what the Skill does and when to use it.' }}
                </div>

                <label>
                  <span>{{ t('name') }}</span>
                  <el-input :model-value="manifest.name" @update:model-value="updateManifest('name', $event)" />
                </label>
                <label>
                  <span>{{ t('version') }}</span>
                  <el-input :model-value="manifest.version" @update:model-value="updateManifest('version', $event)" />
                </label>
                <label>
                  <span class="label-action-row">
                    <span>{{ t('description') }}</span>
                    <span class="inline-actions">
                      <el-button size="small" @click="aiPolishManifestDescription">{{ t('aiPolish') }}</el-button>
                    </span>
                  </span>
                  <el-input
                    :model-value="manifest.description"
                    type="textarea"
                    :autosize="{ minRows: 3, maxRows: 8 }"
                    @update:model-value="updateManifest('description', $event)"
                  />
                </label>
              </section>

              <section v-else-if="detailGroup === 'manifest' && detailSection === 'derived'" class="panel-section">
                <div class="section-title">{{ t('workflowDerived') }}</div>
                <div class="helper-text">
                  {{ isChinese ? '这些内容由当前工作流自动生成，不再作为独立表单维护。导出 Skill 时会使用这里的派生结果。' : 'These values are generated from the current workflow instead of maintained as separate form data. Skill export uses this derived view.' }}
                </div>

                <div class="subsection-title">{{ t('derivedTags') }}</div>
                <div class="field-pill-list">
                  <span v-for="tag in derivedTags" :key="tag" class="field-pill">{{ tag }}</span>
                  <span v-if="!derivedTags.length" class="field-empty">{{ isChinese ? '无' : 'None' }}</span>
                </div>

                <div class="subsection-title">{{ t('derivedTools') }}</div>
                <div v-if="derivedTools.length" class="derived-tool-list">
                  <div v-for="tool in derivedTools" :key="`${tool.binding}:${tool.name}`" class="derived-tool-row">
                    <div>
                      <strong>{{ tool.name }}</strong>
                      <span>{{ tool.binding }}</span>
                    </div>
                    <p>{{ tool.description || (isChinese ? '无描述' : 'No description') }}</p>
                    <small>{{ isChinese ? '来源' : 'Source' }}: {{ tool.source }}</small>
                  </div>
                </div>
                <div v-else class="field-empty">{{ isChinese ? '当前工作流没有声明工具或逻辑节点。' : 'The current workflow has no declared tools or logic nodes.' }}</div>
              </section>

              <section v-else-if="detailGroup === 'manifest' && detailSection === 'behavior'" class="panel-section">
                <div class="label-action-row">
                  <div class="section-title">{{ t('globalRules') }}</div>
                  <span class="inline-actions">
                    <el-button size="small" @click="aiPolishManifestBehavior">{{ t('aiPolish') }}</el-button>
                  </span>
                </div>
                <div class="helper-text">
                  {{ isChinese ? '这里仅维护 Skill 级全局规则。节点自身规则仍在对应推理节点设置中维护，并会在导出时按工作流合并。' : 'This section only maintains Skill-level global rules. Node rules stay with their reasoning node settings and are merged from the workflow during export.' }}
                </div>
                <BehaviorEditor
                  :value="manifest.behavior || { style: '', rules: [] }"
                  :help="isChinese ? '在大文本框中直接编辑，每行一条规则。下面的建议可一键追加。' : 'Edit directly in the large text box. Use one rule per line. Suggestions below can be appended with one click.'"
                  :rule-suggestions="behaviorRuleSuggestions"
                  @update:value="updateManifest('behavior', $event)"
                />
              </section>

              <section v-else-if="detailGroup === 'agent' && detailSection === 'basics' && selectedNode?.type === 'agent'" class="panel-section">
                <div class="section-title">{{ t('agentBasics') }}</div>
                <label>
                  <span>{{ t('label') }}</span>
                  <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
                </label>
                <label>
                  <span>{{ t('type') }}</span>
                  <el-input :model-value="selectedNode.type" disabled />
                </label>
              </section>

              <section v-else-if="detailGroup === 'agent' && detailSection === 'prompt' && selectedNode?.type === 'agent'" class="panel-section">
                <div class="section-title">{{ t('agentPrompt') }}</div>
                <label>
                  <span class="label-action-row">
                    <span>{{ t('instructions') }}</span>
                    <span class="inline-actions">
                      <el-button size="small" @click="aiPolishNodeConfig('instructions')">{{ t('aiPolish') }}</el-button>
                    </span>
                  </span>
                  <el-input
                    :model-value="selectedNode.config.instructions || ''"
                    type="textarea"
                    :autosize="{ minRows: 4, maxRows: 10 }"
                    @update:model-value="updateNodeConfig('instructions', $event)"
                  />
                </label>

                <label>
                  <span class="label-action-row">
                    <span>{{ t('promptTemplate') }}</span>
                    <span class="inline-actions">
                      <el-button size="small" @click="aiPolishNodeConfig('prompt_template')">{{ t('aiPolish') }}</el-button>
                    </span>
                  </span>
                  <el-input
                    :model-value="selectedNode.config.prompt_template || ''"
                    type="textarea"
                    :autosize="{ minRows: 4, maxRows: 12 }"
                    @update:model-value="updateNodeConfig('prompt_template', $event)"
                  />
                </label>

                <div v-if="agentPromptWarnings.length" class="warning-block">
                  <div class="warning-title">{{ isChinese ? '提示词警告' : 'Prompt Warnings' }}</div>
                  <div v-for="warning in agentPromptWarnings" :key="warning" class="field-help warning-text">
                    {{ warning }}
                  </div>
                </div>
              </section>

              <section v-else-if="detailGroup === 'agent' && detailSection === 'keys' && selectedNode?.type === 'agent'" class="panel-section">
                <div class="section-title">{{ t('agentKeys') }}</div>
                <div class="subsection-title">{{ t('pullKeys') }}</div>
                <MapEditor
                  :value="selectedNode.config.pull_keys || {}"
                  :key-label="t('inputKey')"
                  :value-label="t('meaning')"
                  key-placeholder="query"
                  :value-placeholder="isChinese ? '原始用户请求' : 'Original user request'"
                  :help="isChinese ? '读取字段声明该节点期望读取哪些字段。' : 'Pull keys tell this node what fields it expects to read.'"
                  :suggestions="agentPullSuggestions"
                  :suggestion-title="t('workflowKeyPool')"
                  key-input-mode="select"
                  :show-suggestion-strip="false"
                  @update:value="updateNodeConfig('pull_keys', $event)"
                />

                <div class="subsection-title">{{ t('pushKeys') }}</div>
                <MapEditor
                  :value="selectedNode.config.push_keys || {}"
                  :key-label="t('outputKey')"
                  :value-label="t('meaning')"
                  key-placeholder="analysis"
                  :value-placeholder="isChinese ? '结构化任务分析' : 'Structured task analysis'"
                  :help="isChinese ? '写入字段声明该节点会为下游节点写回哪些字段。' : 'Push keys tell this node what fields it writes back for downstream nodes.'"
                  :suggestions="agentPushSuggestions"
                  :suggestion-title="t('workflowKeyPool')"
                  key-input-mode="select"
                  :show-suggestion-strip="false"
                  @update:value="updateNodeConfig('push_keys', $event)"
                />
              </section>

              <section v-else-if="detailGroup === 'agent' && detailSection === 'tools' && selectedNode?.type === 'agent'" class="panel-section">
                <div class="section-title">{{ t('nodeTools') }}</div>
                <div class="helper-text">
                  {{ isChinese ? '这些工具是这一阶段可以使用的工具。行为要求请直接写在提示词或指令中。' : 'These are the tools this stage can use. Put behavior requirements directly in the prompt or instructions.' }}
                </div>
                <ToolListEditor
                  :value="selectedNode.config.tools || []"
                  :title="t('nodeTools')"
                  :help="isChinese ? '这些工具只作用于当前推理阶段。' : 'These tools only apply to this reasoning stage.'"
                  @update:value="updateNodeConfig('tools', $event)"
                />
              </section>

              <section v-else-if="detailGroup === 'custom' && detailSection === 'basics' && selectedNode?.type === 'custom'" class="panel-section">
                <div class="section-title">{{ t('customNodeBasics') }}</div>
                <label>
                  <span>{{ t('label') }}</span>
                  <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
                </label>
                <label>
                  <span>{{ t('type') }}</span>
                  <el-input :model-value="selectedNode.type" disabled />
                </label>
                <el-button type="primary" @click="$emit('open-custom-editor', selectedNode.id)">{{ t('openCustomEditor') }}</el-button>
              </section>

              <section v-else-if="detailGroup === 'custom' && detailSection === 'runtime' && selectedNode?.type === 'custom'" class="panel-section">
                <div class="section-title">{{ t('customRuntimeView') }}</div>
                <div class="field-help">
                  {{ t('pullKeys') }}: <code>{{ Object.keys(selectedNode.config.pull_keys || {}).join(', ') || (isChinese ? '无' : 'none') }}</code>
                </div>
                <div class="field-help">
                  {{ t('pushKeys') }}: <code>{{ Object.keys(selectedNode.config.push_keys || {}).join(', ') || (isChinese ? '无' : 'none') }}</code>
                </div>
              </section>

              <section v-else-if="detailGroup === 'loop' && detailSection === 'basics' && selectedNode?.type === 'loop'" class="panel-section">
                <div class="section-title">{{ t('loopBasics') }}</div>
                <label>
                  <span>{{ t('label') }}</span>
                  <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
                </label>
                <el-button type="primary" @click="$emit('open-loop-editor', selectedNode.id)">{{ t('openLoopEditor') }}</el-button>
              </section>

              <section v-else-if="detailGroup === 'loop' && detailSection === 'termination' && selectedNode?.type === 'loop'" class="panel-section">
                <div class="section-title">{{ t('loopTermination') }}</div>
                <label>
                  <span>{{ t('maxIterations') }}</span>
                  <el-input-number
                    class="number-input"
                    :min="1"
                    :model-value="selectedNode.config.max_iterations || 3"
                    @update:model-value="updateNodeConfig('max_iterations', Number($event) || 1)"
                  />
                </label>

                <label>
                  <span>{{ t('mode') }}</span>
                  <el-select
                    :model-value="selectedNode.config.terminate_when?.mode || 'never'"
                    @update:model-value="updateNestedConfig('terminate_when', 'mode', $event)"
                  >
                    <el-option v-for="option in terminateModeOptions" :key="option.value" :value="option.value" :label="option.label" />
                  </el-select>
                </label>
                <label v-if="(selectedNode.config.terminate_when?.mode || 'never') !== 'never'">
                  <span>{{ t('key') }}</span>
                  <el-input
                    :model-value="selectedNode.config.terminate_when?.key || ''"
                    @update:model-value="updateNestedConfig('terminate_when', 'key', $event)"
                  />
                </label>
                <label v-if="selectedNode.config.terminate_when?.mode === 'key_equals'">
                  <span>{{ t('value') }}</span>
                  <el-input
                    :model-value="selectedNode.config.terminate_when?.value ?? true"
                    @update:model-value="updateNestedConfig('terminate_when', 'value', $event)"
                  />
                </label>
              </section>

              <section v-else-if="detailGroup === 'loop' && detailSection === 'structure' && selectedNode?.type === 'loop'" class="panel-section">
                <div class="section-title">{{ t('loopStructure') }}</div>
                <div class="field-help">
                  {{ isChinese ? 'Loop 内部结构在专用编辑器中编辑。' : 'Loop internals live in a dedicated editor.' }}
                </div>
                <el-button type="primary" @click="$emit('open-loop-editor', selectedNode.id)">{{ t('openLoopEditor') }}</el-button>
              </section>

              <section v-else-if="detailGroup === 'node' && detailSection === 'basics' && selectedNode" class="panel-section">
                <div class="section-title">{{ t('nodeBasics') }}</div>
                <label>
                  <span>{{ t('label') }}</span>
                  <el-input :model-value="selectedNode.label" @update:model-value="updateNodeField('label', $event)" />
                </label>
                <label>
                  <span>{{ t('type') }}</span>
                  <el-input :model-value="selectedNode.type" disabled />
                </label>
              </section>
          </main>
        </div>
      </div>
    </Teleport>
  </div>
</template>
