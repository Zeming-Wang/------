<script setup>
import { computed, ref, watch } from 'vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  value: { type: Object, default: () => ({}) },
  keyLabel: { type: String, default: 'Key' },
  valueLabel: { type: String, default: 'Value' },
  keyPlaceholder: { type: String, default: 'key' },
  valuePlaceholder: { type: String, default: 'value' },
  help: { type: String, default: '' },
  suggestions: { type: Array, default: () => [] },
  suggestionTitle: { type: String, default: 'Suggested Keys' },
  suggestionValueMode: { type: String, default: 'description' },
  keyInputMode: { type: String, default: 'input' },
  showSuggestionStrip: { type: Boolean, default: true }
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

function createRow(key = '', value = '') {
  return {
    id: nextRowId(),
    key,
    value
  };
}

function normalizeEntries(raw) {
  return Object.entries(raw || {}).map(([key, value]) => ({
    key: String(key),
    value: String(value ?? '')
  }));
}

function serializeEntries(entries) {
  return JSON.stringify(entries.map((entry) => [entry.key, entry.value]));
}

function syncRows(raw) {
  const normalized = normalizeEntries(raw);
  const nextRows = normalized.map((entry, index) => ({
    id: rows.value[index]?.id || nextRowId(),
    key: entry.key,
    value: entry.value
  }));
  const draftRows = rows.value
    .slice(normalized.length)
    .filter((row) => !String(row.key || '').trim() && !String(row.value || '').trim())
    .map((row) => ({ ...row }));
  rows.value = [...nextRows, ...draftRows];
  if (!rows.value.length) {
    rows.value = [createRow()];
  }
}

watch(
  () => props.value,
  (nextValue) => {
    const signature = serializeEntries(normalizeEntries(nextValue));
    if (signature === lastEmittedSignature.value) return;
    syncRows(nextValue);
    lastEmittedSignature.value = signature;
  },
  { deep: true, immediate: true }
);

const normalizedSuggestions = computed(() =>
  (props.suggestions || [])
    .map((item) =>
      typeof item === 'string'
        ? { key: item, value: '' }
        : { key: String(item?.key || ''), value: String(item?.value || '') }
    )
    .filter((item) => item.key)
);

function commitFromRows(nextRows) {
  const out = {};
  for (const row of nextRows) {
    const key = String(row.key || '').trim();
    if (!key) continue;
    out[key] = String(row.value ?? '');
  }
  lastEmittedSignature.value = serializeEntries(normalizeEntries(out));
  emit('update:value', out);
}

function updateRow(index, patch) {
  rows.value[index] = {
    ...rows.value[index],
    ...patch
  };
  commitFromRows(rows.value);
}

function addRow() {
  rows.value.push(createRow());
}

function removeRow(index) {
  rows.value.splice(index, 1);
  if (!rows.value.length) {
    rows.value.push(createRow());
  }
  commitFromRows(rows.value);
}

function applySuggestion(suggestion) {
  const suggestionValue =
    props.suggestionValueMode === 'key' ? suggestion.key : suggestion.value || '';
  const matchIndex = rows.value.findIndex((row) => String(row.key || '').trim() === suggestion.key);
  if (matchIndex >= 0) {
    if (!String(rows.value[matchIndex].value || '').trim() && suggestionValue) {
      rows.value[matchIndex] = {
        ...rows.value[matchIndex],
        value: suggestionValue
      };
      commitFromRows(rows.value);
    }
    return;
  }
  rows.value.push(createRow(suggestion.key, suggestionValue));
  commitFromRows(rows.value);
}
</script>

<template>
  <div class="structured-editor">
    <div v-if="help" class="field-help">{{ help }}</div>
    <div v-if="showSuggestionStrip && normalizedSuggestions.length" class="suggestion-strip">
      <div class="suggestion-title">{{ suggestionTitle }}</div>
      <div class="suggestion-list">
        <el-button
          v-for="suggestion in normalizedSuggestions"
          :key="suggestion.key"
          class="suggestion-chip"
          text
          @click="applySuggestion(suggestion)"
        >
          <span>{{ suggestion.key }}</span>
          <small v-if="suggestion.value">{{ suggestion.value }}</small>
        </el-button>
      </div>
    </div>
    <div class="row-head row-grid map-grid">
      <div>{{ keyLabel }}</div>
      <div>{{ valueLabel }}</div>
      <div></div>
    </div>
    <div class="editor-rows">
      <div v-for="(row, index) in rows" :key="row.id" class="row-grid map-grid">
        <el-select
          v-if="keyInputMode === 'select'"
          :model-value="row.key"
          filterable
          allow-create
          default-first-option
          clearable
          :placeholder="keyPlaceholder"
          @update:model-value="updateRow(index, { key: $event })"
        >
          <el-option
            v-for="suggestion in normalizedSuggestions"
            :key="suggestion.key"
            :value="suggestion.key"
            :label="suggestion.value ? `${suggestion.key} - ${suggestion.value}` : suggestion.key"
          />
        </el-select>
        <el-input
          v-else
          :model-value="row.key"
          :placeholder="keyPlaceholder"
          @update:model-value="updateRow(index, { key: $event })"
        />
        <el-input
          :model-value="row.value"
          :placeholder="valuePlaceholder"
          @update:model-value="updateRow(index, { value: $event })"
        />
        <el-button class="mini-button" @click="removeRow(index)">×</el-button>
      </div>
    </div>
    <el-button type="primary" class="mini-button" @click="addRow">{{ t('addRow') }}</el-button>
  </div>
</template>
