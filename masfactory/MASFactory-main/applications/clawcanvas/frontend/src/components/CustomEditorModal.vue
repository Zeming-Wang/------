<script setup>
import { computed, ref } from 'vue';
import MapEditor from './MapEditor.vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  customNode: { type: Object, required: true },
  keyPool: { type: Object, required: true }
});

const emit = defineEmits(['close', 'rename-custom', 'update-custom-config']);
const { isChinese, t } = useI18n();
const activeTab = ref('basics');

const modeOptions = computed(() => [
  { value: 'python', label: isChinese.value ? '逻辑代码' : 'Logic Code' },
  { value: 'compose', label: isChinese.value ? '组合输出' : 'Compose Mixed Output' },
  { value: 'template', label: t('templates') },
  { value: 'set', label: isChinese.value ? '设置静态值' : 'Set Static Values' },
  { value: 'pick', label: t('pickKeys') },
  { value: 'passthrough', label: isChinese.value ? '透传' : 'Passthrough' }
]);

const editorTabs = computed(() => [
  { id: 'basics', label: t('basics') },
  { id: 'keys', label: t('keys') },
  ...(['template', 'compose'].includes(props.customNode.config?.mode || 'python')
    ? [{ id: 'templates', label: t('templates') }]
    : []),
  ...(['set', 'compose'].includes(props.customNode.config?.mode || 'python')
    ? [{ id: 'static', label: t('staticOutputs') }]
    : []),
  ...(['pick', 'compose'].includes(props.customNode.config?.mode || 'python')
    ? [{ id: 'pick', label: t('pickKeys') }]
    : []),
  { id: 'code', label: t('pythonLogic') }
]);

const starterPython = `def forward(input_dict, attributes):
    """
    input_dict: current edge message payload
    attributes: workflow-level / node-level visible attributes
    return: dict payload for downstream edges
    """
    query = input_dict.get("query") or input_dict.get("message", "")
    return {
        "message": f"processed: {query}",
        "length": len(str(query)),
    }
`;

const globalSuggestions = computed(() =>
  Object.entries(props.keyPool?.key_map || {}).map(([key, value]) => ({
    key: String(key),
    value: String(value ?? '')
  }))
);

function updateConfig(field, value) {
  emit('update-custom-config', {
    id: props.customNode.id,
    config: {
      ...props.customNode.config,
      [field]: value
    }
  });
}

function ensurePythonStarter() {
  if (String(props.customNode.config?.python_code || '').trim()) return;
  updateConfig('python_code', starterPython);
}

function updateMode(value) {
  updateConfig('mode', value);
  if (value === 'python') {
    ensurePythonStarter();
  }
}
</script>

<template>
  <div class="loop-modal-backdrop" @click.self="$emit('close')">
    <div class="loop-modal-shell custom-modal-shell">
      <header class="loop-modal-header">
        <div>
          <div class="eyebrow">{{ t('customNodeEditor') }}</div>
          <h2>{{ customNode.label }}</h2>
          <p>{{ t('customEditorHelp') }}</p>
        </div>
        <el-button @click="$emit('close')">{{ t('close') }}</el-button>
      </header>

      <div class="inspector-modal-tabs custom-editor-tabs">
        <el-tabs v-model="activeTab" class="settings-tabs">
          <el-tab-pane
            v-for="tab in editorTabs"
            :key="tab.id"
            :name="tab.id"
            :label="tab.label"
          />
        </el-tabs>

        <section v-if="activeTab === 'basics'" class="panel-section">
            <div class="section-title">{{ t('nodeIdentity') }}</div>
            <label>
              <span>{{ t('label') }}</span>
              <el-input :model-value="customNode.label" @update:model-value="$emit('rename-custom', { id: customNode.id, label: $event })" />
            </label>
            <label>
              <span>{{ t('mode') }}</span>
              <el-select :model-value="customNode.config.mode || 'python'" @update:model-value="updateMode($event)">
                <el-option v-for="option in modeOptions" :key="option.value" :value="option.value" :label="option.label" />
              </el-select>
            </label>
            <div class="field-help">
              {{ t('pythonModeHelp') }}
            </div>
          </section>

          <section v-else-if="activeTab === 'keys'" class="panel-section custom-editor-section">
            <div class="section-title">{{ t('nodeKeys') }}</div>
            <div class="subsection-title">{{ t('pullKeys') }}</div>
            <MapEditor
              :value="customNode.config.pull_keys || {}"
              :key-label="t('inputKey')"
              :value-label="t('meaning')"
              key-placeholder="query"
              :value-placeholder="isChinese ? '逻辑代码可读取的输入' : 'Input available to logic code'"
              :help="isChinese ? '这些字段会出现在 input_dict 中，也可从外层属性作用域读取。' : 'These fields are available in input_dict and outer attribute scope.'"
              :suggestions="globalSuggestions"
              :suggestion-title="t('workflowKeyPool')"
              key-input-mode="select"
              :show-suggestion-strip="false"
              @update:value="updateConfig('pull_keys', $event)"
            />
            <div class="subsection-title">{{ t('pushKeys') }}</div>
            <MapEditor
              :value="customNode.config.push_keys || {}"
              :key-label="t('outputKey')"
              :value-label="t('meaning')"
              key-placeholder="result"
              :value-placeholder="isChinese ? '逻辑代码生成的输出' : 'Output produced by logic code'"
              :help="isChinese ? 'forward 函数通常应返回这些字段，供下游节点使用。' : 'Your forward function should usually return these keys for downstream nodes.'"
              :suggestions="globalSuggestions"
              :suggestion-title="t('workflowKeyPool')"
              key-input-mode="select"
              :show-suggestion-strip="false"
              @update:value="updateConfig('push_keys', $event)"
            />
          </section>

          <section v-else-if="activeTab === 'templates'" class="panel-section custom-editor-section">
            <div class="section-title">{{ t('templates') }}</div>
            <MapEditor
              :value="customNode.config.templates || {}"
              :key-label="t('outputKey')"
              :value-label="t('templates')"
              key-placeholder="summary"
              value-placeholder="Summary: {message}"
              :suggestions="globalSuggestions"
              :suggestion-title="t('workflowKeyPool')"
              key-input-mode="select"
              :show-suggestion-strip="false"
              @update:value="updateConfig('templates', $event)"
            />
          </section>

          <section v-else-if="activeTab === 'static'" class="panel-section custom-editor-section">
            <div class="section-title">{{ t('staticOutputs') }}</div>
            <MapEditor
              :value="customNode.config.static_outputs || {}"
              :key-label="t('outputKey')"
              :value-label="t('value')"
              key-placeholder="status"
              value-placeholder="ready"
              :suggestions="globalSuggestions"
              :suggestion-title="t('workflowKeyPool')"
              key-input-mode="select"
              :show-suggestion-strip="false"
              @update:value="updateConfig('static_outputs', $event)"
            />
          </section>

          <section v-else-if="activeTab === 'pick'" class="panel-section custom-editor-section">
            <div class="section-title">{{ t('pickKeys') }}</div>
            <MapEditor
              :value="customNode.config.pick_keys || {}"
              :key-label="t('outputKey')"
              :value-label="isChinese ? '来源字段' : 'Source Key'"
              key-placeholder="copied_query"
              value-placeholder="query"
              :suggestions="globalSuggestions"
              :suggestion-title="t('workflowKeyPool')"
              suggestion-value-mode="key"
              key-input-mode="select"
              :show-suggestion-strip="false"
              @update:value="updateConfig('pick_keys', $event)"
            />
          </section>

          <section v-else class="panel-section custom-code-panel">
            <div class="section-title">{{ t('pythonLogic') }}</div>
            <div class="field-help">
              {{ t('pythonLogicHelp') }}
            </div>
            <pre class="code-hint">def forward(input_dict, attributes) -> dict:</pre>
            <div class="field-help">
              {{ t('pythonInputHelp') }}
            </div>
            <el-input
              class="custom-code-editor"
              :model-value="customNode.config.python_code ?? starterPython"
              type="textarea"
              resize="vertical"
              spellcheck="false"
              @update:model-value="updateConfig('python_code', $event)"
            />
          </section>
      </div>
    </div>
  </div>
</template>
