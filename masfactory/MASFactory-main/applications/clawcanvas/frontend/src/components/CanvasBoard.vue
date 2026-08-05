<script setup>
import { computed, nextTick, onBeforeUnmount, ref } from 'vue';
import { useI18n } from '../composables/useI18n';

const NODE_WIDTH = 180;
const WORLD_WIDTH = 5000;
const WORLD_HEIGHT = 3200;
const WORLD_ORIGIN_X = 1000;
const WORLD_ORIGIN_Y = 700;
const DRAG_CLICK_THRESHOLD = 3;

const props = defineProps({
  nodes: { type: Array, required: true },
  edges: { type: Array, required: true },
  selectedNodeId: { type: String, default: '' },
  selectedEdgeId: { type: String, default: '' }
});

const emit = defineEmits([
  'select-node',
  'select-edge',
  'move-node',
  'create-edge',
  'rename-node',
  'open-loop',
  'status'
]);
const { isChinese, t } = useI18n();

const viewportRef = ref(null);
const boardRef = ref(null);
const labelInputRef = ref(null);

const pan = ref({ x: 64 - WORLD_ORIGIN_X, y: 64 - WORLD_ORIGIN_Y });
const pointerWorld = ref({ x: 0, y: 0 });
const renameDraft = ref('');
const editingNodeId = ref('');
const dragPreview = ref(null);

let dragState = null;
let panState = null;
let pendingConnection = null;
let dragFrame = 0;

const nodesById = computed(() => {
  const map = new Map();
  for (const node of displayNodes.value) {
    map.set(node.id, node);
  }
  return map;
});

const displayNodes = computed(() => {
  if (!dragPreview.value) return props.nodes;
  return props.nodes.map((node) => {
    if (node.id !== dragPreview.value.id) return node;
    return {
      ...node,
      position: dragPreview.value.position
    };
  });
});

const worldBounds = {
  width: WORLD_WIDTH,
  height: WORLD_HEIGHT
};

function boardX(x) {
  return x + WORLD_ORIGIN_X;
}

function boardY(y) {
  return y + WORLD_ORIGIN_Y;
}

const renderedEdges = computed(() => {
  return props.edges
    .map((edge) => {
      const source = nodesById.value.get(edge.source);
      const target = nodesById.value.get(edge.target);
      if (!source || !target) return null;
      return {
        ...edge,
        x1: boardX(source.position.x + NODE_WIDTH),
        y1: boardY(source.position.y + 34),
        x2: boardX(target.position.x),
        y2: boardY(target.position.y + 34)
      };
    })
    .filter(Boolean);
});

const previewEdge = computed(() => {
  if (!pendingConnection) return null;
  const source = nodesById.value.get(pendingConnection.nodeId);
  if (!source) return null;
  const startX =
    pendingConnection.side === 'right'
      ? boardX(source.position.x + NODE_WIDTH)
      : boardX(source.position.x);
  const startY = boardY(source.position.y + 34);
  return {
    x1: startX,
    y1: startY,
    x2: pointerWorld.value.x,
    y2: pointerWorld.value.y
  };
});

function nodeHandleRole(node, side) {
  if (node.meta?.handleMode?.[side]) {
    return node.meta.handleMode[side];
  }
  if (side === 'left') {
    return node.type === 'start' ? 'disabled' : 'input';
  }
  return node.type === 'end' ? 'disabled' : 'output';
}

function onNodeMouseDown(event, node) {
  if (editingNodeId.value) return;
  if (node.meta?.locked) {
    emit('select-node', node.id);
    return;
  }
  dragState = {
    nodeId: node.id,
    startX: event.clientX,
    startY: event.clientY,
    originX: node.position.x,
    originY: node.position.y,
    dx: 0,
    dy: 0,
    element: event.currentTarget,
    moved: false
  };
  dragState.element?.classList.add('dragging');
  window.addEventListener('mousemove', onWindowMouseMove);
  window.addEventListener('mouseup', onWindowMouseUp);
}

function onViewportMouseDown(event) {
  const isCanvasBackground =
    event.target === viewportRef.value ||
    event.target === boardRef.value ||
    event.target?.classList?.contains?.('canvas-lines');
  if (!isCanvasBackground) return;
  panState = {
    startX: event.clientX,
    startY: event.clientY,
    originX: pan.value.x,
    originY: pan.value.y
  };
  emit('select-node', '');
  window.addEventListener('mousemove', onWindowMouseMove);
  window.addEventListener('mouseup', onWindowMouseUp);
}

function onWindowMouseMove(event) {
  if (pendingConnection) {
    updatePointerWorld(event.clientX, event.clientY);
  }

  if (dragState) {
    scheduleDragPreview(event);
  }

  if (panState) {
    const dx = event.clientX - panState.startX;
    const dy = event.clientY - panState.startY;
    pan.value = {
      x: panState.originX + dx,
      y: panState.originY + dy
    };
  }
}

function onWindowMouseUp(event) {
  commitDrag(event);
  dragState = null;
  panState = null;
  window.removeEventListener('mousemove', onWindowMouseMove);
  window.removeEventListener('mouseup', onWindowMouseUp);
}

function onViewportMouseMove(event) {
  if (!pendingConnection) return;
  updatePointerWorld(event.clientX, event.clientY);
}

function positionFromDragEvent(event = null) {
  const dx = event ? event.clientX - dragState.startX : dragState.dx;
  const dy = event ? event.clientY - dragState.startY : dragState.dy;
  return {
    id: dragState.nodeId,
    position: {
      x: dragState.originX + dx,
      y: dragState.originY + dy
    }
  };
}

function scheduleDragPreview(event) {
  dragState.dx = event.clientX - dragState.startX;
  dragState.dy = event.clientY - dragState.startY;
  dragState.moved =
    Math.abs(dragState.dx) > DRAG_CLICK_THRESHOLD ||
    Math.abs(dragState.dy) > DRAG_CLICK_THRESHOLD;
  if (dragFrame) return;
  dragFrame = window.requestAnimationFrame(() => {
    dragFrame = 0;
    if (!dragState) return;
    dragPreview.value = {
      id: dragState.nodeId,
      position: {
        x: dragState.originX + dragState.dx,
        y: dragState.originY + dragState.dy
      }
    };
  });
}

function commitDrag(event) {
  if (!dragState) return;
  const state = dragState;
  if (event) {
    state.dx = event.clientX - state.startX;
    state.dy = event.clientY - state.startY;
  }
  if (dragFrame) {
    window.cancelAnimationFrame(dragFrame);
    dragFrame = 0;
  }
  const finalMove = positionFromDragEvent(event);
  if (state.element) {
    state.element.classList.remove('dragging');
  }
  const moved =
    Math.abs(state.dx) > DRAG_CLICK_THRESHOLD ||
    Math.abs(state.dy) > DRAG_CLICK_THRESHOLD;
  if (moved && finalMove) emit('move-node', finalMove);
  dragPreview.value = null;
  if (!moved) emit('select-node', state.nodeId);
}

function updatePointerWorld(clientX, clientY) {
  const rect = viewportRef.value?.getBoundingClientRect();
  if (!rect) return;
  pointerWorld.value = {
    x: clientX - rect.left - pan.value.x,
    y: clientY - rect.top - pan.value.y
  };
}

function clickHandle(node, side) {
  if (editingNodeId.value) return;
  const role = nodeHandleRole(node, side);
  if (role === 'disabled') return;

  emit('select-node', node.id);

  if (!pendingConnection) {
    pendingConnection = { nodeId: node.id, side, role };
    pointerWorld.value = {
      x: side === 'right' ? boardX(node.position.x + NODE_WIDTH) : boardX(node.position.x),
      y: boardY(node.position.y + 34)
    };
    emit(
      'status',
      isChinese.value
        ? `已选择 ${node.id} 的${side === 'right' ? '右侧' : '左侧'}手柄。请点击另一个兼容手柄创建连线。`
        : `Selected ${node.id} ${side} handle. Click another compatible handle to create an edge.`
    );
    return;
  }

  const first = pendingConnection;
  if (first.nodeId === node.id && first.side === side) {
    pendingConnection = null;
    emit('status', isChinese.value ? '已取消连线。' : 'Connection cancelled.');
    return;
  }

  if (first.role === role) {
    pendingConnection = { nodeId: node.id, side, role };
    emit(
      'status',
      isChinese.value
        ? `已重新选择 ${node.id} 的${side === 'right' ? '右侧' : '左侧'}手柄。现在请点击兼容手柄。`
        : `Re-selected ${node.id} ${side} handle. Now click a compatible handle.`
    );
    return;
  }

  const source = first.role === 'output' ? first : { nodeId: node.id, side, role };
  const target = first.role === 'input' ? first : { nodeId: node.id, side, role };

  if (source.nodeId === target.nodeId) {
    emit('status', isChinese.value ? '源节点和目标节点不能相同。' : 'Source and target cannot be the same node.');
    pendingConnection = null;
    return;
  }

  emit('create-edge', { source: source.nodeId, target: target.nodeId });
  pendingConnection = null;
}

function onViewportClick(event) {
  const isCanvasBackground =
    event.target === viewportRef.value ||
    event.target === boardRef.value ||
    event.target?.classList?.contains?.('canvas-lines');
  if (isCanvasBackground) {
    if (pendingConnection) {
      pendingConnection = null;
      emit('status', isChinese.value ? '已取消连线。' : 'Connection cancelled.');
    }
    emit('select-node', '');
    emit('select-edge', '');
  }
}

function selectEdge(edgeId) {
  if (editingNodeId.value) return;
  pendingConnection = null;
  emit('select-edge', edgeId);
}

function beginRename(node) {
  if (node.meta?.renamable === false) return;
  emit('select-node', node.id);
  editingNodeId.value = node.id;
  renameDraft.value = node.label;
  nextTick(() => {
    const input = Array.isArray(labelInputRef.value)
      ? labelInputRef.value.find(Boolean)
      : labelInputRef.value;
    input?.focus?.();
    input?.select?.();
  });
}

function commitRename() {
  if (!editingNodeId.value) return;
  const trimmed = renameDraft.value.trim();
  if (trimmed) {
    emit('rename-node', { id: editingNodeId.value, label: trimmed });
  }
  editingNodeId.value = '';
  renameDraft.value = '';
}

function cancelRename() {
  editingNodeId.value = '';
  renameDraft.value = '';
}

function canOpenLoop(node) {
  return node.type === 'loop' || Boolean(node.meta?.openLoop);
}

function handleNodeDoubleClick(node) {
  if (canOpenLoop(node)) {
    emit('open-loop', node.id);
    return;
  }
  beginRename(node);
}

onBeforeUnmount(() => {
  if (dragFrame) {
    window.cancelAnimationFrame(dragFrame);
    dragFrame = 0;
  }
  if (dragState?.element) {
    dragState.element.classList.remove('dragging');
  }
  dragPreview.value = null;
  window.removeEventListener('mousemove', onWindowMouseMove);
  window.removeEventListener('mouseup', onWindowMouseUp);
});
</script>

<template>
  <div
    ref="viewportRef"
    class="canvas-viewport"
    @mousedown="onViewportMouseDown"
    @mousemove="onViewportMouseMove"
    @click="onViewportClick"
  >
    <div
      ref="boardRef"
      class="canvas-board"
      :style="{
        width: `${worldBounds.width}px`,
        height: `${worldBounds.height}px`,
        transform: `translate(${pan.x}px, ${pan.y}px)`
      }"
    >
      <svg class="canvas-lines" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <marker
            id="arrow"
            markerWidth="12"
            markerHeight="12"
            refX="10"
            refY="6"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M 0 0 L 12 6 L 0 12 z" fill="#29425f" />
          </marker>
        </defs>
        <template v-for="edge in renderedEdges" :key="edge.id">
          <line
            :x1="edge.x1"
            :y1="edge.y1"
            :x2="edge.x2"
            :y2="edge.y2"
            class="edge-hit-line"
            @mousedown.stop
            @click.stop="selectEdge(edge.id)"
          />
          <line
            :x1="edge.x1"
            :y1="edge.y1"
            :x2="edge.x2"
            :y2="edge.y2"
            class="edge-line"
            :class="{ selected: edge.id === selectedEdgeId }"
            marker-end="url(#arrow)"
          />
        </template>
        <line
          v-if="previewEdge"
          :x1="previewEdge.x1"
          :y1="previewEdge.y1"
          :x2="previewEdge.x2"
          :y2="previewEdge.y2"
          class="edge-line preview"
          marker-end="url(#arrow)"
        />
      </svg>

      <div
        v-for="node in displayNodes"
        :key="node.id"
        class="canvas-node"
        :class="[
          `type-${node.type}`,
          { selected: node.id === selectedNodeId, renaming: editingNodeId === node.id }
        ]"
        :style="{ left: `${boardX(node.position.x)}px`, top: `${boardY(node.position.y)}px` }"
        @mousedown.stop="onNodeMouseDown($event, node)"
        @dblclick.stop="handleNodeDoubleClick(node)"
      >
        <button
          type="button"
          class="node-handle input-handle"
          :class="{ disabled: nodeHandleRole(node, 'left') === 'disabled' }"
          @mousedown.stop
          @click.stop="clickHandle(node, 'left')"
        />
        <button
          type="button"
          class="node-handle output-handle"
          :class="{ disabled: nodeHandleRole(node, 'right') === 'disabled' }"
          @mousedown.stop
          @click.stop="clickHandle(node, 'right')"
        />
        <div class="node-type">{{ node.meta?.displayType || node.type }}</div>
        <div v-if="editingNodeId !== node.id" class="node-label" @dblclick.stop="handleNodeDoubleClick(node)">
          {{ node.label }}
        </div>
        <input
          v-else
          ref="labelInputRef"
          v-model="renameDraft"
          class="node-label-input"
          @mousedown.stop
          @click.stop
          @keydown.enter.prevent="commitRename"
          @keydown.esc.prevent="cancelRename"
          @blur="commitRename"
        />
        <div class="node-id">{{ node.id }}</div>
        <button
          v-if="canOpenLoop(node)"
          type="button"
          class="mini-button loop-open-button"
          @mousedown.stop
          @click.stop="emit('open-loop', node.id)"
        >
          {{ t('editLoop') }}
        </button>
      </div>
    </div>
  </div>
</template>
