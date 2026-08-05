<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { ElMessage } from 'element-plus';
import CanvasBoard from './components/CanvasBoard.vue';
import CustomEditorModal from './components/CustomEditorModal.vue';
import InspectorPanel from './components/InspectorPanel.vue';
import LoopEditorModal from './components/LoopEditorModal.vue';
import MapEditor from './components/MapEditor.vue';
import { useI18n } from './composables/useI18n';
import { buildNodeTemplate, createDemoDocument, createEmptyDocument, nextNodeId, normalizeLoopConfig } from './composables/useClawCanvas';

const DEFAULT_API_ROOT = '/api';
const API_ROOT = String(import.meta.env.VITE_API_ROOT || DEFAULT_API_ROOT).replace(/\/$/, '');
const AUTO_COLLAPSE_WIDTH = 1180;

const { isChinese, locale, setLanguage, t } = useI18n();
const documentRef = ref(createEmptyDocument());
const selectedNodeId = ref('');
const selectedEdgeId = ref('');
const apiKey = ref('');
const baseUrl = ref('');
const modelName = ref('gpt-4o-mini');
const authoringApiKey = ref('');
const authoringBaseUrl = ref('');
const authoringModelName = ref('gpt-4o-mini');
const statusText = ref('Idle');
const validationSummary = ref(null);
const runResult = ref(null);
const warnings = ref([]);
const keyPool = ref({ keys: [], key_names: [], key_map: {} });
const loopEditorNodeId = ref('');
const customEditorNodeId = ref('');
const settingsModal = ref('');
const leftCollapsed = ref(false);
const rightCollapsed = ref(false);
const autoCompact = ref(false);
const activeResultTab = ref('validation');
const resultPanelHeight = ref(260);
let resultResizeState = null;
let autoValidateTimer = null;
let lastAutoValidationSignature = '';

function notify(message, type = 'success') {
  statusText.value = message;
  ElMessage({
    message,
    type,
    grouping: true,
    duration: 2600
  });
}

const shellClasses = computed(() => ({
  'left-collapsed': leftCollapsed.value,
  'right-collapsed': rightCollapsed.value,
  'auto-compact': autoCompact.value
}));

const selectedNode = computed(() =>
  documentRef.value.nodes.find((node) => node.id === selectedNodeId.value) || null
);

const selectedEdge = computed(() =>
  documentRef.value.edges.find((edge) => edge.id === selectedEdgeId.value) || null
);

const selectedEdgeSuggestions = computed(() =>
  selectedEdge.value ? mappingSuggestionsForEdge(selectedEdge.value) : []
);

const nodesById = computed(() => {
  const map = new Map();
  for (const node of documentRef.value.nodes) {
    map.set(node.id, node);
  }
  return map;
});

const selectedLoopNode = computed(() =>
  documentRef.value.nodes.find((node) => node.id === loopEditorNodeId.value && node.type === 'loop') || null
);

const selectedCustomNode = computed(() =>
  documentRef.value.nodes.find((node) => node.id === customEditorNodeId.value && node.type === 'custom') || null
);

const settingsTitle = computed(() => {
  if (settingsModal.value === 'runtime') return t('runtimeAndExport');
  if (settingsModal.value === 'authoringAi') return t('authoringAiSettings');
  return t('document');
});

const resultTabs = computed(() => [
  {
    id: 'validation',
    label: t('validation'),
    help: t('validationHelp'),
    content: formatValidationResult()
  },
  {
    id: 'run',
    label: t('runOutput'),
    help: t('runOutputHelp'),
    content: formatRunResult()
  },
  {
    id: 'warnings',
    label: t('warnings'),
    help: t('warningsHelp'),
    content: warnings.value.length
      ? warnings.value.map((item) => `- ${humanizeIssue(item)}`).join('\n')
      : t('noWarnings')
  }
]);

const activeResult = computed(() =>
  resultTabs.value.find((item) => item.id === activeResultTab.value) || resultTabs.value[0]
);

const languageOptions = computed(() => [
  { label: 'EN', value: 'en' },
  { label: '中文', value: 'zh' }
]);

function updateResponsiveShell() {
  if (typeof window === 'undefined') return;
  const shouldCompact = window.innerWidth < AUTO_COLLAPSE_WIDTH;
  autoCompact.value = shouldCompact;
  if (shouldCompact) {
    leftCollapsed.value = true;
    rightCollapsed.value = true;
  }
  resultPanelHeight.value = clampResultPanelHeight(resultPanelHeight.value);
}

function toggleLeftPanel() {
  leftCollapsed.value = !leftCollapsed.value;
}

function toggleRightPanel() {
  rightCollapsed.value = !rightCollapsed.value;
}

function clampResultPanelHeight(value) {
  const viewportHeight = typeof window === 'undefined' ? 900 : window.innerHeight;
  const maxHeight = Math.max(220, Math.floor(viewportHeight * 0.52));
  return Math.min(Math.max(Number(value) || 260, 180), maxHeight);
}

function onResultResizeMove(event) {
  if (!resultResizeState) return;
  const delta = resultResizeState.startY - event.clientY;
  resultPanelHeight.value = clampResultPanelHeight(resultResizeState.startHeight + delta);
}

function stopResultResize() {
  resultResizeState = null;
  window.removeEventListener('pointermove', onResultResizeMove);
  window.removeEventListener('pointerup', stopResultResize);
}

function startResultResize(event) {
  resultResizeState = {
    startY: event.clientY,
    startHeight: resultPanelHeight.value
  };
  window.addEventListener('pointermove', onResultResizeMove);
  window.addEventListener('pointerup', stopResultResize);
}

function stringifyValue(value) {
  if (value === null || value === undefined || value === '') return isChinese.value ? '空' : 'empty';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function formatMapBlock(title, value, emptyText) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return `${title}\n- result: ${stringifyValue(value)}`;
  }
  const entries = Object.entries(value || {});
  if (!entries.length) return `${title}\n- ${emptyText}`;
  return `${title}\n${entries.map(([key, item]) => `- ${key}: ${stringifyValue(item)}`).join('\n')}`;
}

function humanizeIssue(rawIssue) {
  const text = typeof rawIssue === 'string' ? rawIssue : String(rawIssue?.message || rawIssue || '');
  if (!isChinese.value) return text;
  const canonical = text.replace(/^[A-Za-z]+Error:\s*/, '');

  if (canonical === 'workflow must contain exactly one start node') {
    return '当前画布需要且只能有一个 Start 节点。';
  }
  if (canonical === 'workflow must contain exactly one end node') {
    return '当前画布需要且只能有一个 End 节点。';
  }
  if (canonical === 'workflow end node is not reachable from start') {
    return '从 Start 节点无法到达 End 节点，请检查节点连线是否完整。';
  }
  if (canonical === 'start node must not have incoming edges') {
    return 'Start 节点不应有进入它的连线。';
  }
  if (canonical === 'end node must not have outgoing edges') {
    return 'End 节点不应有离开它的连线。';
  }

  let match = canonical.match(/^(.+): duplicate node id '([^']+)'/);
  if (match) return `${match[1]} 中存在重复节点 ID：${match[2]}。`;

  match = canonical.match(/^(.+): duplicate edge id '([^']+)'/);
  if (match) return `${match[1]} 中存在重复连线 ID：${match[2]}。`;

  match = canonical.match(/^edge '([^']+)' mapping must not be empty/);
  if (match) return `连线 ${match[1]} 的字段映射不能为空。`;

  match = canonical.match(/^edge '([^']+)' mapping contains an empty field name/);
  if (match) return `连线 ${match[1]} 的字段映射里存在空字段名。`;

  match = canonical.match(/^edge '([^']+)' references an unknown node/);
  if (match) return `连线 ${match[1]} 引用了不存在的节点。`;

  match = canonical.match(/^edge '([^']+)' must not target the start node/);
  if (match) return `连线 ${match[1]} 不应连接到 Start 节点作为目标。`;

  match = canonical.match(/^edge '([^']+)' must not leave the end node/);
  if (match) return `连线 ${match[1]} 不应从 End 节点出发。`;

  match = canonical.match(/^(?:agent|reasoning) node '([^']+)' must define instructions/);
  if (match) return `推理节点 ${match[1]} 缺少指令内容。`;

  match = canonical.match(/^(?:custom|logic) node '([^']+)' uses (?:python|logic code) mode but python_code is empty/);
  if (match) return `逻辑节点 ${match[1]} 缺少逻辑代码。`;

  match = canonical.match(/^(?:custom|logic) node '([^']+)' has unsupported mode '([^']+)'/);
  if (match) return `逻辑节点 ${match[1]} 使用了不支持的模式：${match[2]}。`;

  match = canonical.match(/^edge '([^']+)' maps entry keys (.+), but document\.inputs only provides (.+)$/);
  if (match) {
    return `连线 ${match[1]} 引用了入口字段 ${match[2]}，但文档输入只提供 ${match[3]}。`;
  }

  match = canonical.match(/^(.+): placeholder '\{([^}]+)\}' appears in (.+) but is not declared in pull_keys/);
  if (match) {
    return `节点 ${match[1]} 的提示词使用了 {${match[2]}}，但该字段没有在读取字段中声明。`;
  }

  match = canonical.match(/^(.+): placeholder '\{([^}]+)\}' appears in (.+) but cannot be found/);
  if (match) {
    return `节点 ${match[1]} 的提示词使用了 {${match[2]}}，但当前工作流里找不到这个字段。`;
  }

  match = canonical.match(/^(.+): loop has no controller inputs/);
  if (match) return `Loop 节点 ${match[1]} 没有控制器输入，外层数据无法进入内部流程。`;

  match = canonical.match(/^(.+): loop has no controller outputs/);
  if (match) return `Loop 节点 ${match[1]} 没有控制器输出，内部结果无法返回外层流程。`;

  match = canonical.match(/^(.+): terminate key '([^']+)' is not present/);
  if (match) return `Loop 节点 ${match[1]} 的终止字段 ${match[2]} 没有出现在控制器输出中，循环可能无法正常结束。`;

  match = canonical.match(/^(.+): controller output source '([^']+)' is not reachable/);
  if (match) return `Loop 节点 ${match[1]} 的控制器输出来源 ${match[2]} 无法从控制器输入到达。`;

  match = canonical.match(/^(.+): inner keys (.+) exist only inside the loop/);
  if (match) return `Loop 节点 ${match[1]} 的内部字段 ${match[2]} 没有映射到控制器输出，外层节点无法读取。`;

  return canonical;
}

function edgeIssueMessage(edgeId, fallbackMessage) {
  const edge = documentRef.value.edges.find((item) => item.id === edgeId);
  if (!edge) return fallbackMessage;
  const source = documentRef.value.nodes.find((node) => node.id === edge.source);
  const target = documentRef.value.nodes.find((node) => node.id === edge.target);
  const sourceLabel = source?.label || edge.source;
  const targetLabel = target?.label || edge.target;
  return `节点 ${sourceLabel} 和节点 ${targetLabel} 之间存在问题：${fallbackMessage}`;
}

function normalizeValidationIssues(workflowData, skillValidation = null) {
  const issues = [];
  for (const error of workflowData?.errors || []) {
    const match = String(error).match(/^edge '([^']+)'/);
    const message = humanizeIssue(error);
    issues.push({
      severity: 'error',
      message: match ? edgeIssueMessage(match[1], message) : message
    });
  }
  for (const warning of workflowData?.warnings || []) {
    const match = String(warning).match(/^edge '([^']+)'/);
    const message = humanizeIssue(warning);
    issues.push({
      severity: 'warning',
      message: match ? edgeIssueMessage(match[1], message) : message
    });
  }
  for (const warning of skillValidation?.warnings || []) {
    issues.push({ severity: 'warning', message: humanizeIssue(warning) });
  }
  for (const error of skillValidation?.errors || []) {
    issues.push({ severity: 'error', message: humanizeIssue(error) });
  }
  if (!workflowData?.ok && workflowData?.error && !(workflowData?.cause_chain || []).length) {
    issues.push({ severity: 'error', message: humanizeIssue(workflowData.error) });
  }
  for (const item of workflowData?.cause_chain || []) {
    const text = item?.message || item;
    const message = humanizeIssue(text);
    const match = String(text).match(/^edge '([^']+)'/);
    issues.push({
      severity: 'error',
      message: match ? edgeIssueMessage(match[1], message) : message
    });
  }
  return issues;
}

function workflowSignature(document) {
  function stripPositionsFromNode(node) {
    const config = node?.config || {};
    const nextConfig = { ...config };
    if (nextConfig.controller_layout) {
      delete nextConfig.controller_layout;
    }
    if (nextConfig.subgraph) {
      nextConfig.subgraph = {
        ...nextConfig.subgraph,
        nodes: (nextConfig.subgraph.nodes || []).map(stripPositionsFromNode)
      };
    }
    const { position, ...rest } = node || {};
    return {
      ...rest,
      config: nextConfig
    };
  }

  try {
    return JSON.stringify({
      name: document.name,
      description: document.description,
      manifest: document.manifest,
      inputs: document.inputs,
      attributes: document.attributes,
      nodes: (document.nodes || []).map(stripPositionsFromNode),
      edges: document.edges
    });
  } catch {
    return String(Date.now());
  }
}

function applyValidationState(workflowData, skillValidation = null) {
  const issues = normalizeValidationIssues(workflowData, skillValidation);
  const hasErrors = issues.some((issue) => issue.severity === 'error');
  validationSummary.value = {
    ok: !hasErrors,
    checkedAt: new Date().toISOString(),
    nodeCount: workflowData?.summary?.node_count ?? documentRef.value.nodes.length,
    edgeCount: workflowData?.summary?.edge_count ?? documentRef.value.edges.length,
    issues
  };
  warnings.value = issues.map((issue) => issue.message);
  return issues;
}

function scheduleAutoValidate() {
  if (autoValidateTimer) window.clearTimeout(autoValidateTimer);
  autoValidateTimer = window.setTimeout(async () => {
    const signature = workflowSignature(documentRef.value);
    if (signature === lastAutoValidationSignature) return;
    lastAutoValidationSignature = signature;
    await validateWorkflow({ silent: true, automatic: true });
  }, 700);
}

function formatValidationResult() {
  if (!validationSummary.value) return t('notValidatedYet');
  const issues = validationSummary.value.issues || [];
  const checkedAt = validationSummary.value.checkedAt
    ? new Date(validationSummary.value.checkedAt).toLocaleTimeString()
    : '';
  const header = issues.length === 0
    ? (isChinese.value ? '合法性校验通过。当前未发现问题。' : 'Validity check passed. No issues found.')
    : (isChinese.value ? `发现 ${issues.length} 个问题。` : `${issues.length} issue(s) found.`);
  const details = [
    isChinese.value
      ? `节点：${validationSummary.value.nodeCount ?? 0}，连线：${validationSummary.value.edgeCount ?? 0}`
      : `Nodes: ${validationSummary.value.nodeCount ?? 0}, edges: ${validationSummary.value.edgeCount ?? 0}`,
    checkedAt ? (isChinese.value ? `校验时间：${checkedAt}` : `Checked at: ${checkedAt}`) : ''
  ].filter(Boolean);

  if (!issues.length) return [header, ...details].join('\n');
  return [
    header,
    ...details,
    '',
    ...(issues.map((issue) => `- ${issue.severity === 'error' ? (isChinese.value ? '错误' : 'Error') : (isChinese.value ? '警告' : 'Warning')}：${issue.message}`))
  ].join('\n');
}

function formatRunResult() {
  if (!runResult.value) return t('noRunResultYet');
  if (!runResult.value.ok) {
    const message = runResult.value.error || runResult.value.root_cause || (isChinese.value ? '未知错误' : 'Unknown error');
    return isChinese.value ? `试运行失败：${message}` : `Run failed: ${message}`;
  }

  const blocks = [
    formatMapBlock(isChinese.value ? 'Workflow 输入' : 'Workflow input', runResult.value.input || {}, isChinese.value ? '没有输入。' : 'No input.'),
    '',
    formatMapBlock(isChinese.value ? 'Workflow 输出' : 'Workflow output', runResult.value.output || {}, isChinese.value ? '没有输出。' : 'No output.')
  ];
  if (runResult.value.warnings?.length) {
    blocks.push('', isChinese.value ? '试运行警告' : 'Run warnings');
    blocks.push(...runResult.value.warnings.map((item) => `- ${humanizeIssue(item)}`));
  }
  return blocks.join('\n');
}

function collectKeyPoolFromDocument(document) {
  const keyMap = new Map();

  function addKey(key, description = '', source = 'unknown', owner = 'workflow') {
    const normalizedKey = String(key || '').trim();
    if (!normalizedKey) return;
    const existing = keyMap.get(normalizedKey) || {
      key: normalizedKey,
      description: '',
      sources: [],
      owners: []
    };
    const normalizedDescription = String(description || '').trim();
    if (normalizedDescription && !existing.description) {
      existing.description = normalizedDescription;
    }
    if (source && !existing.sources.includes(source)) existing.sources.push(source);
    if (owner && !existing.owners.includes(owner)) existing.owners.push(owner);
    keyMap.set(normalizedKey, existing);
  }

  for (const [key, value] of Object.entries(document.inputs || {})) {
    addKey(key, value, 'document.inputs', 'workflow');
  }
  for (const [key, value] of Object.entries(document.attributes || {})) {
    addKey(key, value, 'document.attributes', 'workflow');
  }
  for (const [key, value] of Object.entries(document.key_descriptions || {})) {
    addKey(key, value, 'document.key_descriptions', 'workflow');
  }
  for (const edge of document.edges || []) {
    for (const [key, value] of Object.entries(edge.mapping || {})) {
      addKey(key, value, 'edge.mapping', edge.id);
    }
  }
  for (const node of document.nodes || []) {
    collectNodeConfigKeys(node.config || {}, node.id, addKey);
  }

  const keys = [...keyMap.values()].sort((a, b) => a.key.localeCompare(b.key));
  keyPool.value = {
    keys,
    key_names: keys.map((item) => item.key),
    key_map: Object.fromEntries(keys.map((item) => [item.key, item.description]))
  };
}

function collectNodeConfigKeys(config, owner, addKey) {
  for (const fieldName of ['pull_keys', 'push_keys', 'templates', 'static_outputs', 'pick_keys']) {
    for (const [key, value] of Object.entries(config[fieldName] || {})) {
      addKey(key, value, `node.${fieldName}`, owner);
    }
  }
  const terminateWhen = config.terminate_when || {};
  if (terminateWhen.key) {
    addKey(terminateWhen.key, 'Loop terminate condition key', 'node.terminate_when', owner);
  }
  for (const controllerEdge of config.controller_inputs || []) {
    for (const [key, value] of Object.entries(controllerEdge.mapping || {})) {
      addKey(key, value, 'node.controller_inputs', owner);
    }
  }
  for (const controllerEdge of config.controller_outputs || []) {
    for (const [key, value] of Object.entries(controllerEdge.mapping || {})) {
      addKey(key, value, 'node.controller_outputs', owner);
    }
  }
  const subgraph = config.subgraph || {};
  for (const edge of subgraph.edges || []) {
    for (const [key, value] of Object.entries(edge.mapping || {})) {
      addKey(key, value, 'node.subgraph.edge.mapping', `${owner}.${edge.id}`);
    }
  }
  for (const node of subgraph.nodes || []) {
    collectNodeConfigKeys(node.config || {}, `${owner}.${node.id}`, addKey);
  }
}

function addNode(type) {
  const id = nextNodeId(documentRef.value, type);
  const node = buildNodeTemplate(type, id);
  node.position = {
    x: 220 + documentRef.value.nodes.length * 18,
    y: 120 + documentRef.value.nodes.length * 22
  };
  documentRef.value.nodes.push(node);
  selectedNodeId.value = id;
  notify(isChinese.value ? `已添加 ${type} 节点 ${id}` : `Added ${type} node ${id}`);
}

function newBlankCanvas() {
  applyDemoDocument(createEmptyDocument());
  notify(isChinese.value ? '已新建空白画布' : 'Blank canvas created');
}

function applyDemoDocument(document, nextKeyPool = null) {
  documentRef.value = document;
  if (nextKeyPool) {
    keyPool.value = nextKeyPool;
  } else {
    collectKeyPoolFromDocument(documentRef.value);
  }
  selectedNodeId.value = '';
  selectedEdgeId.value = '';
  loopEditorNodeId.value = '';
  customEditorNodeId.value = '';
  validationSummary.value = null;
  runResult.value = null;
  warnings.value = [];
}

async function resetDemo() {
  notify(isChinese.value ? '正在重置示例...' : 'Resetting demo...', 'info');
  try {
    const response = await fetch(`${API_ROOT}/demo`);
    if (!response.ok) {
      throw new Error(`Backend demo request failed: ${response.status}`);
    }
    const data = await response.json();
    applyDemoDocument(data.document, data.key_pool || null);
    notify(isChinese.value ? '示例已从后端重置' : 'Demo reset from backend');
  } catch (error) {
    applyDemoDocument(createDemoDocument());
    warnings.value = [
      isChinese.value
        ? `无法加载后端示例，已回退到前端内置示例。${error.message || ''}`.trim()
        : `Could not load backend demo; restored the built-in frontend demo instead. ${error.message || ''}`.trim()
    ];
    notify(isChinese.value ? '已重置为内置示例' : 'Built-in demo restored', 'warning');
  }
}

function selectNode(nodeId) {
  selectedNodeId.value = nodeId;
  selectedEdgeId.value = '';
}

function selectEdge(edgeId) {
  selectedEdgeId.value = edgeId;
  selectedNodeId.value = '';
}

function moveNode({ id, position }) {
  const node = documentRef.value.nodes.find((item) => item.id === id);
  if (!node) return;
  node.position = position;
}

function createEdge({ source, target }) {
  const sourceNode = documentRef.value.nodes.find((node) => node.id === source);
  const targetNode = documentRef.value.nodes.find((node) => node.id === target);
  if (!sourceNode || !targetNode) return;
  if (documentRef.value.edges.some((edge) => edge.source === source && edge.target === target)) {
    notify(isChinese.value
      ? `连线 ${source} -> ${target} 已存在`
      : `Edge ${source} -> ${target} already exists`, 'warning');
    return;
  }
  const id = `edge_${documentRef.value.edges.length + 1}`;
  documentRef.value.edges.push({
    id,
    source,
    target,
    mapping: defaultEdgeMapping(sourceNode, targetNode)
  });
  notify(isChinese.value
    ? `已创建连线 ${source} -> ${target}`
    : `Created edge ${source} -> ${target}`);
}

function defaultEdgeMapping(sourceNode, targetNode) {
  const sourceKeys = Object.entries(readNodeOutputKeys(sourceNode));
  const targetKeys = Object.entries(readNodeInputKeys(targetNode));
  const sourceKeySet = new Set(sourceKeys.map(([key]) => key));

  const intersection = targetKeys.filter(([key]) => sourceKeySet.has(key));
  if (intersection.length) {
    return Object.fromEntries(intersection.map(([key, value]) => [key, String(value ?? '')]));
  }

  if (sourceKeys.length) {
    const [key, value] = sourceKeys[0];
    return { [key]: String(value ?? '') };
  }

  if (targetKeys.length) {
    const [key, value] = targetKeys[0];
    return { [key]: String(value ?? '') };
  }

  return { message: 'message' };
}

function updateNode({ id, patch }) {
  const index = documentRef.value.nodes.findIndex((item) => item.id === id);
  if (index === -1) return;
  const current = documentRef.value.nodes[index];
  documentRef.value.nodes[index] = {
    ...current,
    ...patch,
    config: patch.config ? { ...current.config, ...patch.config } : current.config
  };
}

function renameNode({ id, label }) {
  const node = documentRef.value.nodes.find((item) => item.id === id);
  if (!node) return;
  node.label = label;
  notify(isChinese.value ? `已将 ${id} 重命名为 ${label}` : `Renamed ${id} to ${label}`);
}

function updateManifest(patch) {
  documentRef.value.manifest = {
    ...documentRef.value.manifest,
    ...patch
  };
  if (Object.prototype.hasOwnProperty.call(patch, 'name')) {
    documentRef.value.name = patch.name;
  }
  if (Object.prototype.hasOwnProperty.call(patch, 'description')) {
    documentRef.value.description = patch.description;
  }
}

function updateDocumentField(field, value) {
  documentRef.value = {
    ...documentRef.value,
    [field]: value,
    manifest:
      field === 'name' || field === 'description'
        ? {
            ...documentRef.value.manifest,
            [field]: value
          }
        : documentRef.value.manifest
  };
}

function updateDocumentMap(field, value) {
  documentRef.value = {
    ...documentRef.value,
    [field]: { ...(value || {}) }
  };
}

function deriveLoopInputKeys(config) {
  const normalized = normalizeLoopConfig(config || {});
  const merged = {};
  for (const item of normalized.controller_inputs || []) {
    Object.assign(merged, item.mapping || {});
  }
  return merged;
}

function deriveLoopOutputKeys(config) {
  const normalized = normalizeLoopConfig(config || {});
  const merged = {};
  for (const item of normalized.controller_outputs || []) {
    Object.assign(merged, item.mapping || {});
  }
  return merged;
}

function deleteNode(nodeId) {
  documentRef.value.nodes = documentRef.value.nodes.filter((node) => node.id !== nodeId);
  documentRef.value.edges = documentRef.value.edges.filter(
    (edge) => edge.source !== nodeId && edge.target !== nodeId
  );
  if (selectedNodeId.value === nodeId) {
    selectedNodeId.value = '';
  }
  if (selectedEdgeId.value) {
    const edgeStillExists = documentRef.value.edges.some((edge) => edge.id === selectedEdgeId.value);
    if (!edgeStillExists) selectedEdgeId.value = '';
  }
  notify(isChinese.value ? `已删除节点 ${nodeId}` : `Deleted node ${nodeId}`);
  if (loopEditorNodeId.value === nodeId) {
    loopEditorNodeId.value = '';
  }
}

function deleteEdge(edgeId) {
  documentRef.value.edges = documentRef.value.edges.filter((edge) => edge.id !== edgeId);
  if (selectedEdgeId.value === edgeId) {
    selectedEdgeId.value = '';
  }
  notify(isChinese.value ? `已删除连线 ${edgeId}` : `Deleted edge ${edgeId}`);
}

function onGlobalKeyDown(event) {
  if (loopEditorNodeId.value) return;
  const target = event.target;
  const tagName = target?.tagName?.toLowerCase?.() || '';
  const isEditable =
    tagName === 'input' ||
    tagName === 'textarea' ||
    Boolean(target?.isContentEditable);
  if (isEditable) return;

  if (event.key !== 'Delete' && event.key !== 'Backspace') return;

  if (selectedEdgeId.value) {
    event.preventDefault();
    deleteEdge(selectedEdgeId.value);
    return;
  }

  if (selectedNodeId.value) {
    const node = documentRef.value.nodes.find((item) => item.id === selectedNodeId.value);
    if (!node || node.type === 'start' || node.type === 'end') return;
    event.preventDefault();
    deleteNode(selectedNodeId.value);
  }
}

onMounted(() => {
  window.addEventListener('keydown', onGlobalKeyDown);
  window.addEventListener('resize', updateResponsiveShell);
  if (typeof window !== 'undefined') {
    resultPanelHeight.value = clampResultPanelHeight(Math.floor(window.innerHeight * 0.32));
  }
  updateResponsiveShell();
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onGlobalKeyDown);
  window.removeEventListener('resize', updateResponsiveShell);
  if (autoValidateTimer) window.clearTimeout(autoValidateTimer);
  stopResultResize();
});

function updateEdgeMapping(edgeId, mapping) {
  const edge = documentRef.value.edges.find((item) => item.id === edgeId);
  if (!edge) return;
  edge.mapping = mapping;
  collectKeyPoolFromDocument(documentRef.value);
}

watch(
  documentRef,
  (nextDocument) => {
    collectKeyPoolFromDocument(nextDocument);
    scheduleAutoValidate();
  },
  { deep: true }
);

function mappingSuggestionsForEdge(edge) {
  const sourceNode = nodesById.value.get(edge.source);
  const targetNode = nodesById.value.get(edge.target);
  const suggestions = new Map();

  for (const [key, value] of Object.entries(readNodeOutputKeys(sourceNode))) {
    suggestions.set(key, { key, value: String(value ?? '') });
  }

  for (const [key, value] of Object.entries(readNodeInputKeys(targetNode))) {
    if (!suggestions.has(key)) {
      suggestions.set(key, { key, value: String(value ?? '') });
    }
  }

  return [...suggestions.values()];
}

function readNodeInputKeys(node) {
  if (!node) return {};
  if (node.type === 'start') return documentRef.value.inputs || {};
  if (node.type === 'agent' || node.type === 'custom') return node.config.pull_keys || {};
  if (node.type === 'loop') {
    return deriveLoopInputKeys(node.config);
  }
  return {};
}

function readNodeOutputKeys(node) {
  if (!node) return {};
  if (node.type === 'start') return documentRef.value.inputs || {};
  if (node.type === 'agent' || node.type === 'custom') return node.config.push_keys || {};
  if (node.type === 'loop') {
    return deriveLoopOutputKeys(node.config);
  }
  return {};
}

function openLoopEditor(nodeId) {
  const node = documentRef.value.nodes.find((item) => item.id === nodeId && item.type === 'loop');
  if (!node) return;
  loopEditorNodeId.value = nodeId;
}

function openCustomEditor(nodeId) {
  const node = documentRef.value.nodes.find((item) => item.id === nodeId && item.type === 'custom');
  if (!node) return;
  customEditorNodeId.value = nodeId;
}

function updateLoopConfig({ id, config }) {
  updateNode({
    id,
    patch: {
      config
    }
  });
}

function updateCustomConfig({ id, config }) {
  updateNode({
    id,
    patch: {
      config
    }
  });
}

async function fetchApi(path, payload) {
  let response;
  try {
    response = await fetch(`${API_ROOT}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  } catch (error) {
    const wrapped = new Error(
      isChinese.value
        ? `无法连接后端 ${API_ROOT}。请启动 Flask 服务并确认 API 端口。`
        : `Cannot reach backend at ${API_ROOT}. Start the Flask server and verify the API port.`
    );
    wrapped.payload = {
      ok: false,
      error: wrapped.message,
      error_type: error?.name || 'NetworkError',
      cause_chain: [
        {
          type: error?.name || 'NetworkError',
          message: error?.message || 'Unknown network error'
        }
      ],
      api_root: API_ROOT,
      request_path: path
    };
    throw wrapped;
  }
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { ok: false, error: text || `Request failed: ${response.status}` };
  }
  if (!response.ok || data.ok === false) {
    const error = new Error(data.error || `Request failed: ${response.status}`);
    error.payload = data;
    throw error;
  }
  return data;
}

function exportDownloadTarget(data) {
  if (data.format === 'skill' && data.skill_zip_path) {
    return { key: 'skill_zip_path', path: data.skill_zip_path, filename: data.skill_zip_path.split(/[\\/]/).pop() || 'skill.zip' };
  }
  if (data.format === 'json' && data.json_path) {
    return { key: 'json_path', path: data.json_path, filename: data.json_path.split(/[\\/]/).pop() || 'clawcanvas.workflow.json' };
  }
  return null;
}

function triggerExportDownload(target) {
  const params = new URLSearchParams({
    key: target.key,
    path: target.path
  });
  const link = document.createElement('a');
  link.href = `${API_ROOT}/download-export?${params.toString()}`;
  link.download = target.filename;
  link.rel = 'noreferrer';
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function validateWorkflow(options = {}) {
  const { silent = false, automatic = false } = options;
  if (!silent) {
    statusText.value = isChinese.value ? '正在进行合法性校验...' : 'Running validity check...';
  }
  try {
    let workflowData;
    try {
      workflowData = await fetchApi('/validate', { document: documentRef.value });
    } catch (error) {
      workflowData = {
        ok: false,
        error: error.payload?.error || error.message,
        cause_chain: error.payload?.cause_chain || [{ message: error.message }],
        summary: {
          node_count: documentRef.value.nodes.length,
          edge_count: documentRef.value.edges.length
        },
        warnings: []
      };
    }
    const skillData = automatic || workflowData.ok === false || workflowData.valid === false
      ? { validation: null }
      : await fetchApi('/validate-skill', {
        document: documentRef.value,
        runOutput: runResult.value || {},
        warnings: warnings.value
      });
    const skillValidation = skillData.validation || {};
    const issues = applyValidationState(workflowData, skillValidation);
    if (workflowData.key_pool) keyPool.value = workflowData.key_pool;
    if (issues.length) {
      notify(isChinese.value
        ? `合法性校验发现 ${issues.length} 个问题`
        : `Validity check found ${issues.length} issue(s)`, issues.some((issue) => issue.severity === 'error') ? 'error' : 'warning');
    } else if (!silent && !automatic) {
      notify(isChinese.value ? '合法性校验通过' : 'Validity check passed');
    }
  } catch (error) {
    const payload = error.payload || {};
    const causeMessages = payload.cause_chain?.length
      ? payload.cause_chain.map((item) => humanizeIssue(`${item.type}: ${item.message}`))
      : [humanizeIssue(payload.error || error.message)];
    validationSummary.value = {
      ok: false,
      checkedAt: new Date().toISOString(),
      nodeCount: documentRef.value.nodes.length,
      edgeCount: documentRef.value.edges.length,
      issues: causeMessages.map((message) => ({ severity: 'error', message }))
    };
    warnings.value = causeMessages;
    notify(isChinese.value ? `校验失败：${error.message}` : `Validation failed: ${error.message}`, 'error');
  }
}

async function runWorkflow() {
  notify(isChinese.value ? '正在运行工作流...' : 'Running workflow...', 'info');
  try {
    const data = await fetchApi('/run', {
      document: documentRef.value,
      runtime: {
        apiKey: apiKey.value,
        baseUrl: baseUrl.value,
        modelName: modelName.value
      }
    });
    runResult.value = {
      ok: true,
      executedAt: new Date().toISOString(),
      input: { ...(documentRef.value.inputs || {}) },
      output: data.output,
      attributes: data.attributes,
      warnings: data.warnings || [],
      runtime: data.runtime || {}
    };
    warnings.value = data.warnings || [];
    notify(isChinese.value
      ? `工作流运行完成（${(data.warnings || []).length} 条警告）`
      : `Workflow run completed (${(data.warnings || []).length} warnings)`, (data.warnings || []).length ? 'warning' : 'success');
  } catch (error) {
    const payload = error.payload || {};
    runResult.value = {
      ok: false,
      executedAt: new Date().toISOString(),
      input: { ...(documentRef.value.inputs || {}) },
      output: {},
      ...payload,
      error: payload.error || error.message
    };
    warnings.value = payload.cause_chain?.length ? payload.cause_chain.map((item) => `${item.type}: ${item.message}`) : [error.message];
    notify(isChinese.value ? `运行失败：${error.message}` : `Run failed: ${error.message}`, 'error');
  }
}

async function aiAuthorField(field, currentValue, applyValue, mode = 'generate') {
  notify(isChinese.value ? '正在调用 AI 润色 Skill 字段...' : 'Polishing Skill field with AI...', 'info');
  try {
    const data = await fetchApi('/ai-authoring/field', {
      document: documentRef.value,
      field,
      mode,
      currentValue,
      locale: locale.value,
      ai: {
        apiKey: authoringApiKey.value,
        baseUrl: authoringBaseUrl.value,
        modelName: authoringModelName.value
      }
    });
    applyValue(data.value);
    notify(isChinese.value ? 'AI 润色内容已应用' : 'AI-polished content applied');
  } catch (error) {
    notify(isChinese.value ? `AI 润色失败：${error.message}` : `AI polish failed: ${error.message}`, 'error');
  }
}

async function exportSkill() {
  notify(isChinese.value ? '正在导出 Skill...' : 'Exporting Skill...', 'info');
  try {
    const data = await fetchApi('/export-skill', {
      document: documentRef.value,
      runOutput: runResult.value || {},
      warnings: warnings.value,
      runtime: {}
    });

    const downloadTarget = exportDownloadTarget(data);
    if (downloadTarget) {
      triggerExportDownload(downloadTarget);
      notify(isChinese.value
        ? `Skill 已导出，正在下载 ${downloadTarget.filename}`
        : `Skill exported; downloading ${downloadTarget.filename}`);
      return;
    }

    notify(isChinese.value
      ? 'Skill 已导出，但没有可下载文件。'
      : 'Skill exported, but no downloadable file was returned.', 'warning');
  } catch (error) {
    notify(error.message, 'error');
  }
}

async function exportJson() {
  notify(isChinese.value ? '正在导出 JSON...' : 'Exporting JSON...', 'info');
  try {
    const data = await fetchApi('/export-json', {
      document: documentRef.value,
      runOutput: runResult.value || {},
      warnings: warnings.value
    });

    const downloadTarget = exportDownloadTarget(data);
    if (downloadTarget) {
      triggerExportDownload(downloadTarget);
      notify(isChinese.value
        ? `JSON 已导出，正在下载 ${downloadTarget.filename}`
        : `JSON exported; downloading ${downloadTarget.filename}`);
      return;
    }

    notify(isChinese.value
      ? 'JSON 已导出，但没有可下载文件。'
      : 'JSON exported, but no downloadable file was returned.', 'warning');
  } catch (error) {
    notify(error.message, 'error');
  }
}

collectKeyPoolFromDocument(documentRef.value);
</script>

<template>
  <div class="app-shell" :class="shellClasses">
    <nav v-if="leftCollapsed" class="sidebar-rail left-rail">
      <el-button
        class="rail-button primary"
        :title="t('expandLeft')"
        @click="toggleLeftPanel"
      >
        ›
      </el-button>
      <el-button class="rail-button" :title="t('addAgent')" @click="addNode('agent')">A</el-button>
      <el-button class="rail-button" :title="t('addCustom')" @click="addNode('custom')">C</el-button>
      <el-button class="rail-button" :title="t('addLoop')" @click="addNode('loop')">L</el-button>
      <el-button class="rail-button" :title="t('newCanvas')" @click="newBlankCanvas">N</el-button>
      <el-button class="rail-button" :title="t('openDocument')" @click="settingsModal = 'document'">D</el-button>
      <el-button class="rail-button" :title="t('openAuthoringAi')" @click="settingsModal = 'authoringAi'">AI</el-button>
      <el-button class="rail-button" :title="t('exportSkill')" @click="exportSkill">↧</el-button>
      <el-button class="rail-button" :title="t('exportJson')" @click="exportJson">J</el-button>
    </nav>

    <aside v-if="!leftCollapsed" class="left-panel">
      <div class="brand">
        <div class="panel-title-row">
          <div class="eyebrow">{{ t('masfactoryApplication') }}</div>
          <el-button
            class="panel-collapse-button"
            :title="t('collapseLeft')"
            @click="toggleLeftPanel"
          >
            ‹
          </el-button>
        </div>
        <h1>ClawCanvas</h1>
        <p>{{ t('appTagline') }}</p>
        <el-segmented
          class="language-switcher"
          :model-value="locale"
          :options="languageOptions"
          size="small"
          :aria-label="t('language')"
          @update:model-value="setLanguage"
        />
      </div>

      <section class="tool-group">
        <div class="group-title">{{ t('canvas') }}</div>
        <el-button @click="newBlankCanvas">{{ t('newCanvas') }}</el-button>
        <el-button type="primary" @click="addNode('agent')">{{ t('addAgent') }}</el-button>
        <el-button type="primary" @click="addNode('custom')">{{ t('addCustom') }}</el-button>
        <el-button type="primary" @click="addNode('loop')">{{ t('addLoop') }}</el-button>
      </section>

      <section class="tool-group">
        <div class="group-title">{{ t('workflowSetup') }}</div>
        <div class="helper-text">{{ t('workflowSetupHelp') }}</div>
        <el-button @click="settingsModal = 'document'">{{ t('openDocument') }}</el-button>
        <el-button @click="settingsModal = 'authoringAi'">{{ t('openAuthoringAi') }}</el-button>
      </section>

      <section class="tool-group">
        <div class="group-title">{{ t('actions') }}</div>
        <el-button type="primary" @click="exportSkill">{{ t('exportSkill') }}</el-button>
        <el-button @click="exportJson">{{ t('exportJson') }}</el-button>
      </section>

      <footer class="masfactory-footer">
        <span>{{ isChinese ? 'ClawCanvas 基于 MASFactory 开发' : 'ClawCanvas is built with MASFactory' }}</span>
        <a href="https://github.com/BUPT-GAMMA/MASFactory" target="_blank" rel="noreferrer">MASFactory GitHub</a>
      </footer>
    </aside>

    <main class="workspace">
      <header class="workspace-header">
        <div>
          <div class="eyebrow">{{ t('workflow') }}</div>
          <h2>{{ documentRef.name }}</h2>
          <p>{{ documentRef.description }}</p>
        </div>
      </header>

      <section class="canvas-stage">
        <CanvasBoard
          :nodes="documentRef.nodes"
          :edges="documentRef.edges"
          :selected-node-id="selectedNodeId"
          :selected-edge-id="selectedEdgeId"
          @select-node="selectNode"
          @select-edge="selectEdge"
          @move-node="moveNode"
          @create-edge="createEdge"
          @rename-node="renameNode"
          @open-loop="openLoopEditor"
          @status="notify($event, 'info')"
        />
      </section>

      <section class="results-panel" :style="{ height: `${resultPanelHeight}px` }">
        <div class="result-resize-handle" @pointerdown.prevent="startResultResize" />
        <article class="console-card result-card">
          <el-tabs v-model="activeResultTab" class="result-tabs">
            <el-tab-pane
              v-for="tab in resultTabs"
              :key="tab.id"
              :name="tab.id"
              :label="tab.label"
            />
          </el-tabs>
          <div class="result-toolbar">
            <div
              v-if="activeResultTab === 'validation' || activeResultTab === 'run'"
              class="result-action-row"
            >
              <el-button v-if="activeResultTab === 'validation'" type="primary" size="small" @click="validateWorkflow">
                {{ t('validityCheck') }}
              </el-button>
              <el-button v-else-if="activeResultTab === 'run'" type="primary" size="small" @click="runWorkflow">
                {{ t('runTest') }}
              </el-button>
            </div>
            <div class="helper-text">{{ activeResult.help }}</div>
          </div>
          <pre>{{ activeResult.content }}</pre>
        </article>
      </section>
    </main>

    <aside v-if="!rightCollapsed" class="right-panel">
      <el-button
        class="panel-collapse-button right-panel-collapse"
        :title="t('collapseRight')"
        @click="toggleRightPanel"
      >
        ›
      </el-button>
      <InspectorPanel
        :selected-node="selectedNode"
        :selected-edge="selectedEdge"
        :edge-suggestions="selectedEdgeSuggestions"
        :manifest="documentRef.manifest"
        :document="documentRef"
        :key-pool="keyPool"
        @update-node="updateNode"
        @update-manifest="updateManifest"
        @update-edge-mapping="updateEdgeMapping"
        @ai-author-field="aiAuthorField"
        @delete-node="deleteNode"
        @delete-edge="deleteEdge"
        @open-loop-editor="openLoopEditor"
        @open-custom-editor="openCustomEditor"
      />
    </aside>

    <nav v-if="rightCollapsed" class="sidebar-rail right-rail">
      <el-button
        class="rail-button primary"
        :title="t('expandRight')"
        @click="toggleRightPanel"
      >
        ‹
      </el-button>
      <el-button class="rail-button" :title="t('openSkillManifest')" @click="rightCollapsed = false">M</el-button>
      <el-button class="rail-button" :title="t('selectedNode')" @click="rightCollapsed = false">N</el-button>
      <el-button class="rail-button" :title="t('editLoop')" @click="selectedNode?.type === 'loop' && openLoopEditor(selectedNode.id)">L</el-button>
      <el-button class="rail-button" :title="t('editCustomNode')" @click="selectedNode?.type === 'custom' && openCustomEditor(selectedNode.id)">C</el-button>
    </nav>

    <LoopEditorModal
      v-if="selectedLoopNode"
      :loop-node="selectedLoopNode"
      :key-pool="keyPool"
      @close="loopEditorNodeId = ''"
      @rename-loop="renameNode"
      @update-loop-config="updateLoopConfig"
    />

    <CustomEditorModal
      v-if="selectedCustomNode"
      :custom-node="selectedCustomNode"
      :key-pool="keyPool"
      @close="customEditorNodeId = ''"
      @rename-custom="renameNode"
      @update-custom-config="updateCustomConfig"
    />

    <div v-if="settingsModal" class="loop-modal-backdrop" @click.self="settingsModal = ''">
      <div
        class="loop-modal-shell settings-modal-shell"
        :class="{ 'runtime-settings-modal-shell': settingsModal === 'runtime' || settingsModal === 'authoringAi' }"
      >
        <header class="loop-modal-header">
          <div>
            <div class="eyebrow">{{ t('workflowSettings') }}</div>
            <h2>{{ settingsTitle }}</h2>
            <p v-if="settingsModal === 'document'">{{ t('documentHelp') }}</p>
            <p v-else-if="settingsModal === 'runtime'">{{ t('runtimeHelp') }}</p>
            <p v-else-if="settingsModal === 'authoringAi'">{{ t('authoringAiHelp') }}</p>
          </div>
          <el-button @click="settingsModal = ''">{{ t('close') }}</el-button>
        </header>

        <div class="settings-modal-body">
          <section v-if="settingsModal === 'document'" class="panel-section">
            <label>
              <span>{{ t('documentName') }}</span>
              <el-input :model-value="documentRef.name" @update:model-value="updateDocumentField('name', $event)" />
            </label>
            <label>
              <span>{{ t('documentDescription') }}</span>
              <el-input
                :model-value="documentRef.description"
                type="textarea"
                :autosize="{ minRows: 3, maxRows: 6 }"
                @update:model-value="updateDocumentField('description', $event)"
              />
            </label>
            <MapEditor
              :value="documentRef.inputs"
              :key-label="t('inputKey')"
              :value-label="t('testValue')"
              key-placeholder="query"
              :value-placeholder="t('workflowEntryValuePlaceholder')"
              :help="t('documentInputsHelp')"
              :suggestions="keyPool.keys"
              :suggestion-title="t('workflowKeyPool')"
              @update:value="updateDocumentMap('inputs', $event)"
            />
            <MapEditor
              :value="documentRef.attributes"
              :key-label="t('attributeKey')"
              :value-label="t('initialValue')"
              key-placeholder="workspace"
              :value-placeholder="t('workflowAttributePlaceholder')"
              :help="t('workflowAttributesHelp')"
              :suggestions="keyPool.keys"
              :suggestion-title="t('workflowKeyPool')"
              @update:value="updateDocumentMap('attributes', $event)"
            />
          </section>

          <section v-else-if="settingsModal === 'runtime'" class="panel-section runtime-settings-section">
            <el-form label-position="left" label-width="108px" class="runtime-settings-form">
              <el-form-item :label="t('apiKey')">
                <el-input v-model="apiKey" size="small" type="password" placeholder="sk-..." show-password />
              </el-form-item>
              <el-form-item :label="t('baseUrl')">
                <el-input
                  v-model="baseUrl"
                  size="small"
                  :placeholder="isChinese ? '可选的 OpenAI 兼容端点' : 'Optional OpenAI-compatible endpoint'"
                />
              </el-form-item>
              <el-form-item :label="t('modelName')">
                <el-input v-model="modelName" size="small" placeholder="gpt-4o-mini" />
              </el-form-item>
            </el-form>
          </section>

          <section v-else-if="settingsModal === 'authoringAi'" class="panel-section runtime-settings-section">
            <el-form label-position="left" label-width="108px" class="runtime-settings-form">
              <el-form-item :label="t('apiKey')">
                <el-input v-model="authoringApiKey" size="small" type="password" placeholder="sk-..." show-password />
              </el-form-item>
              <el-form-item :label="t('baseUrl')">
                <el-input
                  v-model="authoringBaseUrl"
                  size="small"
                  :placeholder="isChinese ? '可选的 OpenAI 兼容端点' : 'Optional OpenAI-compatible endpoint'"
                />
              </el-form-item>
              <el-form-item :label="t('modelName')">
                <el-input v-model="authoringModelName" size="small" placeholder="gpt-4o-mini" />
              </el-form-item>
            </el-form>
          </section>

        </div>
      </div>
    </div>
  </div>
</template>
