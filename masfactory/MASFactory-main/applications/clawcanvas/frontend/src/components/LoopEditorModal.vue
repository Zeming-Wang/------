<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import CanvasBoard from './CanvasBoard.vue';
import LoopNodeInspector from './LoopNodeInspector.vue';
import MapEditor from './MapEditor.vue';
import { useI18n } from '../composables/useI18n';
import { buildNodeTemplate, nextNodeId, normalizeLoopConfig } from '../composables/useClawCanvas';

const CONTROLLER_IN_ID = '__controller_in__';
const CONTROLLER_OUT_ID = '__controller_out__';

const props = defineProps({
  loopNode: { type: Object, required: true },
  keyPool: { type: Object, required: true }
});

const emit = defineEmits(['close', 'update-loop-config', 'rename-loop']);

const selectedNodeId = ref('');
const selectedEdgeId = ref('');
const nestedLoopId = ref('');
const activeSidebarTab = ref('settings');
const activeConnectionTab = ref('nodes');
const loopConfig = ref(normalizeLoopConfig(props.loopNode.config || {}));
const { isChinese, t } = useI18n();

const sidebarTabs = computed(() => [
  { id: 'settings', label: t('loopSettings') },
  { id: 'connections', label: isChinese.value ? '连接' : 'Connections' },
  { id: 'selection', label: selectedNodeId.value || selectedEdgeId.value ? (isChinese.value ? '选中项' : 'Selection') : (isChinese.value ? '编辑' : 'Edit') }
]);
const sidebarTabOptions = computed(() => sidebarTabs.value.map((tab) => ({ label: tab.label, value: tab.id })));

const connectionTabs = computed(() => [
  { id: 'nodes', label: isChinese.value ? '内部节点' : 'Nodes' },
  { id: 'inputs', label: t('controllerInputs') },
  { id: 'outputs', label: t('controllerOutputs') }
]);
const connectionTabOptions = computed(() => connectionTabs.value.map((tab) => ({ label: tab.label, value: tab.id })));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

watch(
  () => props.loopNode.config,
  (config) => {
    loopConfig.value = normalizeLoopConfig(config || {});
  },
  { deep: false }
);

const innerDocument = computed(() => ({
  nodes: loopConfig.value.subgraph.nodes || [],
  edges: [
    ...(loopConfig.value.controller_inputs || []).map((item) => ({
      id: `controller_in:${item.id}`,
      source: CONTROLLER_IN_ID,
      target: item.target,
      mapping: item.mapping || {}
    })),
    ...(loopConfig.value.subgraph.edges || []),
    ...(loopConfig.value.controller_outputs || []).map((item) => ({
      id: `controller_out:${item.id}`,
      source: item.source,
      target: CONTROLLER_OUT_ID,
      mapping: item.mapping || {}
    }))
  ]
}));

const canvasNodes = computed(() => [
  {
    id: CONTROLLER_IN_ID,
    type: 'start',
    label: isChinese.value ? '控制器输入' : 'Controller In',
    position: loopConfig.value.controller_layout?.input_position || { x: 40, y: 220 },
    meta: {
      renamable: false,
      handleMode: { left: 'disabled', right: 'output' },
      displayType: 'controller'
    }
  },
  ...(loopConfig.value.subgraph.nodes || []),
  {
    id: CONTROLLER_OUT_ID,
    type: 'end',
    label: isChinese.value ? '控制器输出' : 'Controller Out',
    position: loopConfig.value.controller_layout?.output_position || { x: 980, y: 220 },
    meta: {
      renamable: false,
      handleMode: { left: 'input', right: 'disabled' },
      displayType: 'controller'
    }
  }
]);

const selectedNode = computed(() =>
  (loopConfig.value.subgraph.nodes || []).find((node) => node.id === selectedNodeId.value) || null
);

const selectedEdge = computed(() =>
  innerDocument.value.edges.find((edge) => edge.id === selectedEdgeId.value) || null
);

const nestedLoopNode = computed(() =>
  (loopConfig.value.subgraph.nodes || []).find((node) => node.id === nestedLoopId.value) || null
);

const innerNodeOptions = computed(() =>
  (loopConfig.value.subgraph.nodes || []).map((node) => ({
    id: node.id,
    label: `${node.label} (${node.id})`
  }))
);

const loopWarnings = computed(() => {
  const warnings = [];
  const controllerInputs = loopConfig.value.controller_inputs || [];
  const controllerOutputs = loopConfig.value.controller_outputs || [];
  const terminate = loopConfig.value.terminate_when || {};

  if (!controllerInputs.length) {
    warnings.push(isChinese.value
      ? 'Loop 没有控制器输入，外层工作流数据不会进入内部图。'
      : 'Loop has no controller inputs, so outer workflow data never enters the inner graph.');
  }
  if (!controllerOutputs.length) {
    warnings.push(isChinese.value
      ? 'Loop 没有控制器输出，内部结果无法返回外层工作流。'
      : 'Loop has no controller outputs, so no inner result can return to the outer workflow.');
  }

  const producedKeys = new Set(
    controllerOutputs.flatMap((item) => Object.keys(item.mapping || {})).map((key) => String(key).trim()).filter(Boolean)
  );
  if ((terminate.mode || 'never') !== 'never' && terminate.key && !producedKeys.has(String(terminate.key))) {
    warnings.push(isChinese.value
      ? `终止字段 "${terminate.key}" 不存在于任何控制器输出映射中。`
      : `Terminate key "${terminate.key}" is not present in any controller output mapping.`);
  }

  const outgoing = new Map();
  for (const node of loopConfig.value.subgraph.nodes || []) {
    outgoing.set(node.id, new Set());
  }
  for (const edge of loopConfig.value.subgraph.edges || []) {
    if (outgoing.has(edge.source) && outgoing.has(edge.target)) {
      outgoing.get(edge.source).add(edge.target);
    }
  }
  const reachable = new Set();
  const stack = controllerInputs.map((item) => item.target).filter((item) => outgoing.has(item));
  while (stack.length) {
    const current = stack.pop();
    if (reachable.has(current)) continue;
    reachable.add(current);
    for (const next of outgoing.get(current) || []) {
      stack.push(next);
    }
  }
  for (const item of controllerOutputs) {
    if (item.source && !reachable.has(item.source)) {
      warnings.push(isChinese.value
        ? `控制器输出源 "${item.source}" 无法从任何控制器输入目标到达。`
        : `Controller output source "${item.source}" is not reachable from any controller input target.`);
    }
  }

  return warnings;
});

const internalOnlyKeys = computed(() => {
  const produced = new Set(
    (loopConfig.value.controller_outputs || [])
      .flatMap((item) => Object.keys(item.mapping || {}))
      .map((key) => String(key).trim())
      .filter(Boolean)
  );
  const inner = new Set();
  for (const node of loopConfig.value.subgraph.nodes || []) {
    for (const key of Object.keys(node.config?.push_keys || {})) {
      if (String(key).trim()) inner.add(String(key).trim());
    }
  }
  return [...inner].filter((key) => !produced.has(key)).sort();
});

function setLoopConfig(nextConfig) {
  emit('update-loop-config', {
    id: props.loopNode.id,
    config: nextConfig
  });
}

function selectNode(nodeId) {
  selectedNodeId.value = nodeId;
  selectedEdgeId.value = '';
  if (nodeId) activeSidebarTab.value = 'selection';
}

function selectEdge(edgeId) {
  selectedEdgeId.value = edgeId;
  selectedNodeId.value = '';
  if (edgeId) activeSidebarTab.value = 'selection';
}

function updateLoopField(field, value) {
  const next = clone(loopConfig.value);
  next[field] = value;
  setLoopConfig(next);
}

function updateTerminateField(field, value) {
  const next = clone(loopConfig.value);
  next.terminate_when = { ...(next.terminate_when || {}), [field]: value };
  setLoopConfig(next);
}

function updateControllerField(field, value) {
  const next = clone(loopConfig.value);
  next.controller = { ...(next.controller || {}), [field]: value };
  setLoopConfig(next);
}

function updateControllerModelField(field, value) {
  const next = clone(loopConfig.value);
  next.controller = {
    ...(next.controller || {}),
    model_settings: {
      ...(next.controller?.model_settings || {}),
      [field]: value
    }
  };
  setLoopConfig(next);
}

function addInnerNode(type) {
  const next = clone(loopConfig.value);
  const id = nextNodeId({ nodes: next.subgraph.nodes || [] }, type);
  const node = buildNodeTemplate(type, id);
  node.position = {
    x: 240 + (next.subgraph.nodes || []).length * 26,
    y: 120 + (next.subgraph.nodes || []).length * 24
  };
  next.subgraph.nodes.push(node);
  setLoopConfig(next);
  selectedNodeId.value = id;
  selectedEdgeId.value = '';
  activeSidebarTab.value = 'selection';
}

function updateInnerNode({ id, patch }) {
  const next = clone(loopConfig.value);
  const index = next.subgraph.nodes.findIndex((node) => node.id === id);
  if (index === -1) return;
  const current = next.subgraph.nodes[index];
  next.subgraph.nodes[index] = {
    ...current,
    ...patch,
    config: patch.config ? { ...current.config, ...patch.config } : current.config
  };
  setLoopConfig(next);
}

function renameInnerNode({ id, label }) {
  const next = clone(loopConfig.value);
  const node = next.subgraph.nodes.find((item) => item.id === id);
  if (!node) return;
  node.label = label;
  setLoopConfig(next);
}

function moveInnerNode({ id, position }) {
  const next = clone(loopConfig.value);
  next.controller_layout = next.controller_layout || {
    input_position: { x: 40, y: 220 },
    output_position: { x: 980, y: 220 }
  };
  if (id === CONTROLLER_IN_ID) {
    next.controller_layout.input_position = position;
    setLoopConfig(next);
    return;
  }
  if (id === CONTROLLER_OUT_ID) {
    next.controller_layout.output_position = position;
    setLoopConfig(next);
    return;
  }
  const node = next.subgraph.nodes.find((item) => item.id === id);
  if (!node) return;
  node.position = position;
  setLoopConfig(next);
}

function deleteInnerNode(nodeId) {
  const next = clone(loopConfig.value);
  next.subgraph.nodes = next.subgraph.nodes.filter((node) => node.id !== nodeId);
  next.subgraph.edges = next.subgraph.edges.filter((edge) => edge.source !== nodeId && edge.target !== nodeId);
  next.controller_inputs = next.controller_inputs.filter((edge) => edge.target !== nodeId);
  next.controller_outputs = next.controller_outputs.filter((edge) => edge.source !== nodeId);
  setLoopConfig(next);
  if (selectedNodeId.value === nodeId) selectedNodeId.value = '';
}

function deleteEdge(edgeId) {
  const next = clone(loopConfig.value);
  if (edgeId.startsWith('controller_in:')) {
    const rawId = edgeId.slice('controller_in:'.length);
    next.controller_inputs = next.controller_inputs.filter((edge) => edge.id !== rawId);
  } else if (edgeId.startsWith('controller_out:')) {
    const rawId = edgeId.slice('controller_out:'.length);
    next.controller_outputs = next.controller_outputs.filter((edge) => edge.id !== rawId);
  } else {
    next.subgraph.edges = next.subgraph.edges.filter((edge) => edge.id !== edgeId);
  }
  setLoopConfig(next);
  if (selectedEdgeId.value === edgeId) selectedEdgeId.value = '';
}

function createEdge({ source, target }) {
  if (source === CONTROLLER_OUT_ID || target === CONTROLLER_IN_ID) return;
  const next = clone(loopConfig.value);
  if (source === CONTROLLER_IN_ID && target !== CONTROLLER_OUT_ID) {
    next.controller_inputs.push({
      id: `controller_in_${next.controller_inputs.length + 1}`,
      target,
      mapping: { message: 'Loop input' }
    });
    setLoopConfig(next);
    return;
  }
  if (target === CONTROLLER_OUT_ID && source !== CONTROLLER_IN_ID) {
    next.controller_outputs.push({
      id: `controller_out_${next.controller_outputs.length + 1}`,
      source,
      mapping: { message: 'Loop output' }
    });
    setLoopConfig(next);
    return;
  }
  if (source !== CONTROLLER_IN_ID && target !== CONTROLLER_OUT_ID) {
    if (next.subgraph.edges.some((edge) => edge.source === source && edge.target === target)) return;
    next.subgraph.edges.push({
      id: `inner_edge_${next.subgraph.edges.length + 1}`,
      source,
      target,
      mapping: { message: 'message' }
    });
    setLoopConfig(next);
  }
}

function updateEdgeMapping(edgeId, mapping) {
  const next = clone(loopConfig.value);
  if (edgeId.startsWith('controller_in:')) {
    const rawId = edgeId.slice('controller_in:'.length);
    const edge = next.controller_inputs.find((item) => item.id === rawId);
    if (!edge) return;
    edge.mapping = mapping;
  } else if (edgeId.startsWith('controller_out:')) {
    const rawId = edgeId.slice('controller_out:'.length);
    const edge = next.controller_outputs.find((item) => item.id === rawId);
    if (!edge) return;
    edge.mapping = mapping;
  } else {
    const edge = next.subgraph.edges.find((item) => item.id === edgeId);
    if (!edge) return;
    edge.mapping = mapping;
  }
  setLoopConfig(next);
}

function controllerDisplay(edge) {
  if (!edge) return '';
  const controllerIn = isChinese.value ? '控制器输入' : 'Controller In';
  const controllerOut = isChinese.value ? '控制器输出' : 'Controller Out';
  if (edge.id.startsWith('controller_in:')) return `${controllerIn} -> ${edge.target}`;
  if (edge.id.startsWith('controller_out:')) return `${edge.source} -> ${controllerOut}`;
  return `${edge.source} -> ${edge.target}`;
}

function mappingSuggestionsForSelectedEdge() {
  if (!selectedEdge.value) return [];
  if (selectedEdge.value.id.startsWith('controller_in:')) {
    const target = (loopConfig.value.subgraph.nodes || []).find((node) => node.id === selectedEdge.value.target);
    return mergeSuggestions(mappingSuggestions(props.keyPool.key_map || {}), mappingSuggestions(target?.config?.pull_keys || {}));
  }
  if (selectedEdge.value.id.startsWith('controller_out:')) {
    const source = (loopConfig.value.subgraph.nodes || []).find((node) => node.id === selectedEdge.value.source);
    return mergeSuggestions(mappingSuggestions(source?.config?.push_keys || {}), mappingSuggestions(props.keyPool.key_map || {}));
  }
  const source = (loopConfig.value.subgraph.nodes || []).find((node) => node.id === selectedEdge.value.source);
  const target = (loopConfig.value.subgraph.nodes || []).find((node) => node.id === selectedEdge.value.target);
  return mergeSuggestions(mappingSuggestions(source?.config?.push_keys || {}), mappingSuggestions(target?.config?.pull_keys || {}));
}

function mappingSuggestions(raw) {
  return Object.entries(raw || {}).map(([key, value]) => ({ key, value: String(value ?? '') }));
}

function mergeSuggestions(...groups) {
  const map = new Map();
  for (const group of groups) {
    for (const item of group || []) {
      const key = String(item?.key || '').trim();
      if (!key) continue;
      if (!map.has(key)) map.set(key, { key, value: String(item?.value || '') });
    }
  }
  return [...map.values()];
}

function openNestedLoop(nodeId) {
  nestedLoopId.value = nodeId;
}

function updateNestedLoopConfig(payload) {
  const next = clone(loopConfig.value);
  const index = next.subgraph.nodes.findIndex((node) => node.id === payload.id);
  if (index === -1) return;
  next.subgraph.nodes[index] = {
    ...next.subgraph.nodes[index],
    config: payload.config
  };
  setLoopConfig(next);
}

function addControllerInput() {
  const target = loopConfig.value.subgraph.nodes?.[0]?.id || '';
  if (!target) return;
  const next = clone(loopConfig.value);
  next.controller_inputs.push({
    id: `controller_in_${next.controller_inputs.length + 1}`,
    target,
    mapping: { message: 'Loop input' }
  });
  setLoopConfig(next);
}

function addControllerOutput() {
  const source = loopConfig.value.subgraph.nodes?.[0]?.id || '';
  if (!source) return;
  const next = clone(loopConfig.value);
  next.controller_outputs.push({
    id: `controller_out_${next.controller_outputs.length + 1}`,
    source,
    mapping: { message: 'Loop output' }
  });
  setLoopConfig(next);
}

function updateControllerInputTarget(id, target) {
  const next = clone(loopConfig.value);
  const edge = next.controller_inputs.find((item) => item.id === id);
  if (!edge) return;
  edge.target = target;
  setLoopConfig(next);
}

function updateControllerOutputSource(id, source) {
  const next = clone(loopConfig.value);
  const edge = next.controller_outputs.find((item) => item.id === id);
  if (!edge) return;
  edge.source = source;
  setLoopConfig(next);
}

function updateControllerInputMapping(id, mapping) {
  const next = clone(loopConfig.value);
  const edge = next.controller_inputs.find((item) => item.id === id);
  if (!edge) return;
  edge.mapping = mapping;
  setLoopConfig(next);
}

function updateControllerOutputMapping(id, mapping) {
  const next = clone(loopConfig.value);
  const edge = next.controller_outputs.find((item) => item.id === id);
  if (!edge) return;
  edge.mapping = mapping;
  setLoopConfig(next);
}

function removeControllerInput(id) {
  const next = clone(loopConfig.value);
  next.controller_inputs = next.controller_inputs.filter((item) => item.id !== id);
  setLoopConfig(next);
  if (selectedEdgeId.value === `controller_in:${id}`) selectedEdgeId.value = '';
}

function removeControllerOutput(id) {
  const next = clone(loopConfig.value);
  next.controller_outputs = next.controller_outputs.filter((item) => item.id !== id);
  setLoopConfig(next);
  if (selectedEdgeId.value === `controller_out:${id}`) selectedEdgeId.value = '';
}

function focusControllerEdge(edgeId, type) {
  selectedNodeId.value = '';
  selectedEdgeId.value = `${type}:${edgeId}`;
  activeSidebarTab.value = 'selection';
}

function onLoopKeyDown(event) {
  const target = event.target;
  const tagName = target?.tagName?.toLowerCase?.() || '';
  const isEditable =
    tagName === 'input' ||
    tagName === 'textarea' ||
    Boolean(target?.isContentEditable);
  if (isEditable) return;

  if (event.key === 'Escape') {
    event.preventDefault();
    event.stopPropagation();
    if (nestedLoopId.value) {
      nestedLoopId.value = '';
      return;
    }
    if (selectedEdgeId.value || selectedNodeId.value) {
      selectedEdgeId.value = '';
      selectedNodeId.value = '';
      return;
    }
    emit('close');
    return;
  }

  if ((event.metaKey || event.ctrlKey) && event.key === '1') {
    event.preventDefault();
    event.stopPropagation();
    addInnerNode('agent');
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key === '2') {
    event.preventDefault();
    event.stopPropagation();
    addInnerNode('custom');
    return;
  }
  if ((event.metaKey || event.ctrlKey) && event.key === '3') {
    event.preventDefault();
    event.stopPropagation();
    addInnerNode('loop');
    return;
  }

  if (event.key !== 'Delete' && event.key !== 'Backspace') return;

  if (selectedEdgeId.value) {
    event.preventDefault();
    event.stopPropagation();
    deleteEdge(selectedEdgeId.value);
    return;
  }

  if (selectedNodeId.value && selectedNodeId.value !== CONTROLLER_IN_ID && selectedNodeId.value !== CONTROLLER_OUT_ID) {
    event.preventDefault();
    event.stopPropagation();
    deleteInnerNode(selectedNodeId.value);
  }
}

onMounted(() => {
  window.addEventListener('keydown', onLoopKeyDown, true);
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onLoopKeyDown, true);
});
</script>

<template>
  <div class="loop-modal-backdrop" @click.self="$emit('close')">
    <div class="loop-modal-shell loop-editor-shell">
      <header class="loop-modal-header">
        <div>
          <div class="eyebrow">{{ t('loopEditor') }}</div>
          <h2>{{ loopNode.label }}</h2>
          <p>{{ t('loopEditorHelp') }}</p>
        </div>
        <el-button @click="$emit('close')">{{ t('close') }}</el-button>
      </header>

      <div class="loop-modal-grid">
        <aside class="loop-modal-sidebar">
          <el-segmented
            v-model="activeSidebarTab"
            class="loop-sidebar-tabs"
            :options="sidebarTabOptions"
          />

          <section v-if="activeSidebarTab === 'settings'" class="tool-group">
            <div class="group-title">{{ t('loopSettings') }}</div>
            <label>
              <span>{{ t('loopLabel') }}</span>
              <el-input :model-value="loopNode.label" @update:model-value="$emit('rename-loop', { id: loopNode.id, label: $event })" />
            </label>
            <label>
              <span>{{ t('maxIterations') }}</span>
              <el-input-number
                class="number-input"
                :min="1"
                :model-value="loopConfig.max_iterations || 3"
                @update:model-value="updateLoopField('max_iterations', Number($event) || 1)"
              />
            </label>
            <label>
              <span>{{ t('controllerTermination') }}</span>
              <el-select :model-value="loopConfig.controller?.termination_mode || 'key_rule'" @update:model-value="updateControllerField('termination_mode', $event)">
                <el-option value="key_rule" :label="t('keyRule')" />
                <el-option value="prompt" :label="t('promptJudge')" />
                <el-option value="expression" :label="t('expression')" />
              </el-select>
            </label>
            <div class="field-help">
              {{ isChinese ? 'Key Rule 是当前简化 UI。Prompt Judge 对应 MASFactory terminate_condition_prompt。Expression 是自定义 terminate_condition_function 的安全序列化替代。' : '`Key Rule` is the current simplified UI. `Prompt Judge` maps to MASFactory terminate_condition_prompt. `Expression` is a safe serialized replacement for a custom terminate_condition_function.' }}
            </div>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'prompt'">
              <span>{{ t('terminateConditionPrompt') }}</span>
              <el-input
                :model-value="loopConfig.controller?.terminate_condition_prompt || ''"
                type="textarea"
                :autosize="{ minRows: 3, maxRows: 8 }"
                @update:model-value="updateControllerField('terminate_condition_prompt', $event)"
              />
            </label>
            <div v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'prompt'" class="field-help">
              {{ isChinese ? '推荐格式：描述 Loop 应停止的精确条件。' : 'Recommended format: describe the exact condition that means the loop should stop.' }}
            </div>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'prompt'">
              <span>{{ t('controllerModelName') }}</span>
              <el-input
                :model-value="loopConfig.controller?.model_settings?.model_name || ''"
                :placeholder="isChinese ? '可选覆盖；否则使用运行时模型' : 'Optional override, otherwise use runtime model'"
                @update:model-value="updateControllerModelField('model_name', $event)"
              />
            </label>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'prompt'">
              <span>{{ t('controllerBaseUrl') }}</span>
              <el-input
                :model-value="loopConfig.controller?.model_settings?.base_url || ''"
                :placeholder="isChinese ? '可选 OpenAI 兼容端点' : 'Optional OpenAI-compatible endpoint'"
                @update:model-value="updateControllerModelField('base_url', $event)"
              />
            </label>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'expression'">
              <span>{{ t('terminateExpression') }}</span>
              <el-input
                :model-value="loopConfig.controller?.terminate_expression || ''"
                type="textarea"
                :autosize="{ minRows: 3, maxRows: 8 }"
                @update:model-value="updateControllerField('terminate_expression', $event)"
              />
            </label>
            <div v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'expression'" class="field-help">
              {{ isChinese ? '使用基于当前 Loop 可见字段的安全布尔表达式。' : 'Use a safe boolean expression over current loop-visible keys.' }}
            </div>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'key_rule'">
              <span>{{ t('terminateMode') }}</span>
              <el-select :model-value="loopConfig.terminate_when?.mode || 'never'" @update:model-value="updateTerminateField('mode', $event)">
                <el-option value="never" :label="t('neverStopEarly')" />
                <el-option value="key_truthy" :label="t('stopWhenKeyTruthy')" />
                <el-option value="key_equals" :label="t('stopWhenKeyEqualsValue')" />
              </el-select>
            </label>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'key_rule' && (loopConfig.terminate_when?.mode || 'never') !== 'never'">
              <span>{{ t('terminateKey') }}</span>
              <el-input :model-value="loopConfig.terminate_when?.key || ''" @update:model-value="updateTerminateField('key', $event)" />
            </label>
            <label v-if="(loopConfig.controller?.termination_mode || 'key_rule') === 'key_rule' && loopConfig.terminate_when?.mode === 'key_equals'">
              <span>{{ t('terminateValue') }}</span>
              <el-input :model-value="loopConfig.terminate_when?.value ?? true" @update:model-value="updateTerminateField('value', $event)" />
            </label>
            <div class="field-help">
              {{ isChinese ? '快捷键：Ctrl/Cmd+1 添加推理节点，Ctrl/Cmd+2 添加逻辑节点，Ctrl/Cmd+3 添加嵌套 Loop，Delete 删除选中连线/节点，Esc 清除选择或关闭编辑器。' : 'Shortcuts: Ctrl/Cmd+1 add reasoning node, Ctrl/Cmd+2 add logic node, Ctrl/Cmd+3 add nested loop, Delete remove selected edge/node, Esc clear selection or close editor.' }}
            </div>
            <div v-if="loopWarnings.length" class="warning-block">
              <div class="warning-title">{{ t('loopWarnings') }}</div>
              <div v-for="warning in loopWarnings" :key="warning" class="field-help warning-text">
                {{ warning }}
              </div>
            </div>
            <div v-if="internalOnlyKeys.length" class="field-help">
              <strong>{{ t('internalKeys') }}:</strong>
              <code>{{ internalOnlyKeys.join(', ') }}</code>
            </div>
          </section>

          <section v-else-if="activeSidebarTab === 'connections'" class="tool-group">
            <el-segmented
              v-model="activeConnectionTab"
              class="inline-tabs compact-tabs"
              :options="connectionTabOptions"
            />

            <template v-if="activeConnectionTab === 'nodes'">
              <div class="group-title">{{ t('innerNodes') }}</div>
              <el-button type="primary" @click="addInnerNode('agent')">{{ t('addAgent') }}</el-button>
              <el-button type="primary" @click="addInnerNode('custom')">{{ t('addCustom') }}</el-button>
              <el-button type="primary" @click="addInnerNode('loop')">{{ t('addLoop') }}</el-button>
              <div class="field-help">
                {{ isChinese ? '将 Controller In 连接到每次迭代应接收输入的第一个内部节点，并将结束节点连接到 Controller Out。' : 'Connect Controller In to the first inner nodes that should receive each iteration input, and connect terminating inner nodes to Controller Out.' }}
              </div>
            </template>

            <template v-else-if="activeConnectionTab === 'inputs'">
              <div class="group-title">{{ t('controllerInputs') }}</div>
              <div class="field-help">
                {{ isChinese ? '这些边对应 MASFactory edge_from_controller，用于把外层 Loop 上下文注入选中的内部节点。' : 'These edges correspond to MASFactory edge_from_controller. They inject outer loop context into selected inner nodes.' }}
              </div>
              <div v-if="!loopConfig.controller_inputs.length" class="field-help">{{ t('noControllerInputs') }}</div>
              <div v-for="item in loopConfig.controller_inputs" :key="item.id" class="object-card">
                <div class="row-head">{{ item.id }}</div>
                <label>
                  <span>{{ t('targetNode') }}</span>
                  <el-select :model-value="item.target" @update:model-value="updateControllerInputTarget(item.id, $event)">
                    <el-option v-for="node in innerNodeOptions" :key="node.id" :value="node.id" :label="node.label" />
                  </el-select>
                </label>
                <MapEditor
                  :value="item.mapping || {}"
                  :key-label="isChinese ? 'Loop 字段' : 'Loop Key'"
                  :value-label="t('meaning')"
                  :suggestions="mergeSuggestions(mappingSuggestions(keyPool.key_map || {}), mappingSuggestions((loopConfig.subgraph.nodes || []).find((node) => node.id === item.target)?.config?.pull_keys || {}))"
                  :suggestion-title="t('workflowKeyPool')"
                  key-input-mode="select"
                  :show-suggestion-strip="false"
                  @update:value="updateControllerInputMapping(item.id, $event)"
                />
                <div class="controller-actions">
                  <el-button class="mini-button" @click="focusControllerEdge(item.id, 'controller_in')">{{ isChinese ? '定位连线' : 'Focus Edge' }}</el-button>
                  <el-button type="danger" class="mini-button" @click="removeControllerInput(item.id)">{{ t('remove') }}</el-button>
                </div>
              </div>
              <el-button type="primary" class="mini-button" @click="addControllerInput">{{ isChinese ? '添加控制器输入' : 'Add Controller Input' }}</el-button>
            </template>

            <template v-else>
              <div class="group-title">{{ t('controllerOutputs') }}</div>
              <div class="field-help">
                {{ isChinese ? '这些边对应 MASFactory edge_to_controller，用于把内部节点结果送回 Loop 控制器和外层工作流。' : 'These edges correspond to MASFactory edge_to_controller. They send inner node results back to the loop controller and outer workflow.' }}
              </div>
              <div v-if="!loopConfig.controller_outputs.length" class="field-help">{{ t('noControllerOutputs') }}</div>
              <div v-for="item in loopConfig.controller_outputs" :key="item.id" class="object-card">
                <div class="row-head">{{ item.id }}</div>
                <label>
                  <span>{{ t('sourceNode') }}</span>
                  <el-select :model-value="item.source" @update:model-value="updateControllerOutputSource(item.id, $event)">
                    <el-option v-for="node in innerNodeOptions" :key="node.id" :value="node.id" :label="node.label" />
                  </el-select>
                </label>
                <MapEditor
                  :value="item.mapping || {}"
                  :key-label="isChinese ? 'Loop 字段' : 'Loop Key'"
                  :value-label="t('meaning')"
                  :suggestions="mergeSuggestions(mappingSuggestions((loopConfig.subgraph.nodes || []).find((node) => node.id === item.source)?.config?.push_keys || {}), mappingSuggestions(keyPool.key_map || {}))"
                  :suggestion-title="t('workflowKeyPool')"
                  key-input-mode="select"
                  :show-suggestion-strip="false"
                  @update:value="updateControllerOutputMapping(item.id, $event)"
                />
                <div class="controller-actions">
                  <el-button class="mini-button" @click="focusControllerEdge(item.id, 'controller_out')">{{ isChinese ? '定位连线' : 'Focus Edge' }}</el-button>
                  <el-button type="danger" class="mini-button" @click="removeControllerOutput(item.id)">{{ t('remove') }}</el-button>
                </div>
              </div>
              <el-button type="primary" class="mini-button" @click="addControllerOutput">{{ isChinese ? '添加控制器输出' : 'Add Controller Output' }}</el-button>
            </template>
          </section>

          <section v-else class="loop-selection-pane">
            <section v-if="selectedEdge" class="tool-group">
              <div class="group-title">{{ isChinese ? '选中连线' : 'Selected Edge' }}</div>
              <div class="field-help">{{ controllerDisplay(selectedEdge) }}</div>
              <MapEditor
                :value="selectedEdge.mapping"
                :key-label="t('key')"
                :value-label="t('meaning')"
                :suggestions="mappingSuggestionsForSelectedEdge()"
                :suggestion-title="t('suggestedKeys')"
                key-input-mode="select"
                :show-suggestion-strip="false"
                @update:value="updateEdgeMapping(selectedEdge.id, $event)"
              />
              <el-button type="danger" @click="deleteEdge(selectedEdge.id)">{{ isChinese ? '删除连线' : 'Delete Edge' }}</el-button>
            </section>

            <LoopNodeInspector
              v-else-if="selectedNode"
              :selected-node="selectedNode"
              :document="innerDocument"
              :key-pool="keyPool"
              @update-node="updateInnerNode"
              @delete-node="deleteInnerNode"
              @open-loop="openNestedLoop"
            />
            <section v-else class="tool-group">
              <div class="group-title">{{ isChinese ? '未选择' : 'No Selection' }}</div>
              <p class="empty-state">
                {{ isChinese ? '点击内部节点或连线后在这里编辑。' : 'Click an inner node or edge to edit it here.' }}
              </p>
            </section>
          </section>
        </aside>

        <main class="loop-modal-main">
          <CanvasBoard
            :nodes="canvasNodes"
            :edges="innerDocument.edges"
            :selected-node-id="selectedNodeId"
            :selected-edge-id="selectedEdgeId"
            @select-node="selectNode"
            @select-edge="selectEdge"
            @move-node="moveInnerNode"
            @create-edge="createEdge"
            @rename-node="renameInnerNode"
            @open-loop="openNestedLoop"
          />
        </main>
      </div>

      <LoopEditorModal
        v-if="nestedLoopNode"
        :loop-node="nestedLoopNode"
        :key-pool="keyPool"
        @close="nestedLoopId = ''"
        @rename-loop="renameInnerNode"
        @update-loop-config="updateNestedLoopConfig"
      />
    </div>
  </div>
</template>
