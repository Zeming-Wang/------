<script setup>
import { ref, watch } from 'vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  value: { type: Array, default: () => [] },
  placeholder: { type: String, default: 'value' },
  help: { type: String, default: '' },
  suggestions: { type: Array, default: () => [] },
  suggestionTitle: { type: String, default: 'Suggested Items' }
});

const emit = defineEmits(['update:value']);
const { t } = useI18n();

const rows = ref([]);
const lastEmittedSignature = ref('');
let rowCounter = 0;

function nextRowId() {
  rowCounter += 1;
  return `row:${rowCounter}`;
}

function createRow(value = '') {
  return {
    id: nextRowId(),
    value
  };
}

function normalizeValue(raw) {
  return (raw || []).map((item) => String(item ?? ''));
}

function serializeValue(raw) {
  return JSON.stringify(raw);
}

function syncRows(raw) {
  const normalized = normalizeValue(raw);
  const nextRows = normalized.map((value, index) => ({
    id: rows.value[index]?.id || nextRowId(),
    value
  }));
  const draftRows = rows.value
    .slice(normalized.length)
    .filter((row) => !String(row.value || '').trim())
    .map((row) => ({ ...row }));
  rows.value = [...nextRows, ...draftRows];
  if (!rows.value.length) {
    rows.value = [createRow()];
  }
}

watch(
  () => props.value,
  (nextValue) => {
    const signature = serializeValue(normalizeValue(nextValue));
    if (signature === lastEmittedSignature.value) return;
    syncRows(nextValue);
    lastEmittedSignature.value = signature;
  },
  { deep: true, immediate: true }
);

function commit() {
  const normalized = rows.value
    .map((row) => String(row.value || '').trim())
    .filter(Boolean);
  lastEmittedSignature.value = serializeValue(normalized);
  emit('update:value', normalized);
}

function updateRow(index, value) {
  rows.value[index] = {
    ...rows.value[index],
    value
  };
  commit();
}

function addRow() {
  rows.value.push(createRow());
}

function removeRow(index) {
  rows.value.splice(index, 1);
  if (!rows.value.length) {
    rows.value.push(createRow());
  }
  commit();
}

function applySuggestion(value) {
  const item = String(value || '').trim();
  if (!item) return;
  const exists = rows.value.some((row) => String(row.value || '').trim() === item);
  if (!exists) {
    rows.value.push(createRow(item));
  }
  commit();
}
</script>

<template>
  <div class="structured-editor">
    <div v-if="help" class="field-help">{{ help }}</div>
    <div v-if="suggestions.length" class="suggestion-strip">
      <div class="suggestion-title">{{ suggestionTitle }}</div>
      <div class="suggestion-list">
        <el-button
          v-for="suggestion in suggestions"
          :key="suggestion"
          class="suggestion-chip"
          text
          @click="applySuggestion(suggestion)"
        >
          <span>{{ suggestion }}</span>
        </el-button>
      </div>
    </div>
    <div class="editor-rows">
      <div v-for="(row, index) in rows" :key="row.id" class="row-grid list-grid">
        <el-input
          :model-value="row.value"
          :placeholder="placeholder"
          @update:model-value="updateRow(index, $event)"
        />
        <el-button class="mini-button" @click="removeRow(index)">×</el-button>
      </div>
    </div>
    <el-button type="primary" class="mini-button" @click="addRow">{{ t('addItem') }}</el-button>
  </div>
</template>
