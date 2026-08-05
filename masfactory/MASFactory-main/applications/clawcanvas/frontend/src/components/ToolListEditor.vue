<script setup>
import { ref, watch } from 'vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  value: { type: Array, default: () => [] },
  help: { type: String, default: '' },
  title: { type: String, default: 'Tools' },
  allowMcp: { type: Boolean, default: true }
});

const emit = defineEmits(['update:value']);
const { t } = useI18n();

const rows = ref([]);
const quickAddValue = ref('');
const lastSignature = ref('');
let rowCounter = 0;

const builtinOptions = [
  {
    value: 'echo',
    label: 'echo',
    help: 'Return input text unchanged.'
  },
  {
    value: 'json_inspect',
    label: 'json_inspect',
    help: 'Render a payload as formatted JSON.'
  },
  {
    value: 'list_keys',
    label: 'list_keys',
    help: 'Return top-level keys from an object payload.'
  },
  {
    value: 'concat_text',
    label: 'concat_text',
    help: 'Concatenate two text fragments.'
  }
];

const httpMethods = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];
const responseModes = ['json', 'text'];
const quickAddOptions = [
  ...builtinOptions.map((item) => ({
    value: `builtin:${item.value}`,
    label: item.label,
    help: item.help
  })),
  {
    value: 'api:http_api',
    label: 'HTTP API',
    help: 'Call an external HTTP endpoint.'
  }
];

function nextRowId() {
  rowCounter += 1;
  return `tool-row:${rowCounter}`;
}

function createDefaultTool() {
  return {
    name: '',
    binding: 'builtin',
    description: '',
    config: {
      method: 'POST',
      url: '',
      headers: {},
      params: {},
      body: {},
      response: 'json',
      timeout: 20
    }
  };
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function normalizeConfig(rawConfig = {}) {
  return {
    method: String(rawConfig.method || 'POST').toUpperCase(),
    url: String(rawConfig.url || ''),
    headers: rawConfig.headers && typeof rawConfig.headers === 'object' ? clone(rawConfig.headers) : {},
    params: rawConfig.params && typeof rawConfig.params === 'object' ? clone(rawConfig.params) : {},
    body: rawConfig.body !== undefined ? clone(rawConfig.body) : {},
    response: String(rawConfig.response || 'json'),
    timeout: rawConfig.timeout !== undefined && rawConfig.timeout !== null && rawConfig.timeout !== '' ? Number(rawConfig.timeout) : 20
  };
}

function normalizeTool(raw) {
  const base = createDefaultTool();
  const item = {
    ...base,
    ...(raw || {}),
    config: normalizeConfig(raw?.config || {})
  };
  if (!item.binding) item.binding = 'builtin';
  return item;
}

function serialize(items) {
  return JSON.stringify(
    items.map((item) => ({
      name: String(item.name || ''),
      binding: String(item.binding || ''),
      description: String(item.description || ''),
      config: normalizeConfig(item.config || {})
    }))
  );
}

function syncRows(rawValue) {
  const normalized = (rawValue || []).map((item) => normalizeTool(item));
  if (normalized.length) {
    rows.value = normalized.map((item, index) => ({
      id: rows.value[index]?.id || nextRowId(),
      data: item
    }));
    return;
  }
  rows.value = [{ id: nextRowId(), data: createDefaultTool() }];
}

watch(
  () => serialize((props.value || []).map((item) => normalizeTool(item))),
  (signature) => {
    if (signature === lastSignature.value) return;
    syncRows(props.value);
    lastSignature.value = signature;
  },
  { immediate: true }
);

function commit() {
  const normalized = rows.value
    .map((row) => normalizeTool(row.data))
    .filter((item) => String(item.name || '').trim() || String(item.description || '').trim());
  lastSignature.value = serialize(normalized);
  emit('update:value', normalized);
}

function updateField(index, field, value) {
  rows.value[index] = {
    ...rows.value[index],
    data: {
      ...rows.value[index].data,
      [field]: value
    }
  };
  commit();
}

function updateConfigField(index, field, value) {
  const current = rows.value[index].data;
  rows.value[index] = {
    ...rows.value[index],
    data: {
      ...current,
      config: {
        ...normalizeConfig(current.config || {}),
        [field]: value
      }
    }
  };
  commit();
}

function parseJsonInput(raw, fallback) {
  const text = String(raw || '').trim();
  if (!text) return clone(fallback);
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === 'object' ? parsed : clone(fallback);
  } catch {
    return raw;
  }
}

function updateJsonConfigField(index, field, raw) {
  const current = rows.value[index].data;
  const fallback = field === 'body' ? {} : {};
  rows.value[index] = {
    ...rows.value[index],
    data: {
      ...current,
      config: {
        ...normalizeConfig(current.config || {}),
        [field]: parseJsonInput(raw, fallback)
      }
    }
  };
  commit();
}

function addTool(preset) {
  rows.value.push({
    id: nextRowId(),
    data: normalizeTool(preset || createDefaultTool())
  });
  commit();
}

function removeTool(index) {
  rows.value.splice(index, 1);
  if (!rows.value.length) {
    rows.value.push({
      id: nextRowId(),
      data: createDefaultTool()
    });
  }
  commit();
}

function selectBuiltin(index, builtinName) {
  const builtin = builtinOptions.find((item) => item.value === builtinName);
  updateField(index, 'name', builtinName);
  if (builtin && !String(rows.value[index].data.description || '').trim()) {
    updateField(index, 'description', builtin.help);
  }
}

function useBuiltinPreset(name) {
  const preset = builtinOptions.find((item) => item.value === name);
  addTool({
    name,
    binding: 'builtin',
    description: preset?.help || '',
    config: createDefaultTool().config
  });
}

function useApiPreset() {
  addTool({
    name: 'http_api',
    binding: 'api',
    description: 'Call an external HTTP endpoint.',
    config: {
      method: 'POST',
      url: 'https://example.com/endpoint',
      headers: {},
      params: {},
      body: { query: '{query}' },
      response: 'json',
      timeout: 20
    }
  });
}

function applyQuickAdd(value) {
  if (!value) return;
  if (value.startsWith('builtin:')) {
    useBuiltinPreset(value.slice('builtin:'.length));
  } else if (value === 'api:http_api') {
    useApiPreset();
  }
  quickAddValue.value = '';
}

function builtinHelp(name) {
  return builtinOptions.find((item) => item.value === name)?.help || t('chooseBuiltinTool');
}
</script>

<template>
  <div class="structured-editor">
    <div class="section-title">{{ title }}</div>
    <div v-if="help" class="field-help">{{ help }}</div>

    <label class="quick-add-select">
      <span>{{ t('quickAdd') }}</span>
      <el-select
        v-model="quickAddValue"
        clearable
        :placeholder="t('quickAdd')"
        @update:model-value="applyQuickAdd"
      >
        <el-option
          v-for="option in quickAddOptions"
          :key="option.value"
          :value="option.value"
          :label="option.label"
        >
          <div class="select-option-rich">
            <strong>{{ option.label }}</strong>
            <small>{{ option.help }}</small>
          </div>
        </el-option>
      </el-select>
    </label>

    <div class="editor-rows">
      <div v-for="(row, index) in rows" :key="row.id" class="object-card">
        <div class="row-head">{{ t('tool') }} {{ index + 1 }}</div>

        <label>
          <span>{{ t('binding') }}</span>
          <el-select :model-value="row.data.binding" @update:model-value="updateField(index, 'binding', $event)">
            <el-option value="builtin" label="builtin" />
            <el-option value="api" label="api" />
            <el-option v-if="allowMcp" value="mcp" label="mcp" />
            <el-option value="other" label="other" />
          </el-select>
        </label>
        <div class="field-help">
          {{ t('builtinBindingHelp') }}
        </div>

        <template v-if="row.data.binding === 'builtin'">
          <label>
            <span>{{ t('builtinTool') }}</span>
            <el-select :model-value="row.data.name" @update:model-value="selectBuiltin(index, $event)">
              <el-option value="" :label="t('select')" />
              <el-option v-for="item in builtinOptions" :key="item.value" :value="item.value" :label="item.label" />
            </el-select>
          </label>
          <div class="field-help">
            {{ builtinHelp(row.data.name) }}
          </div>
        </template>

        <template v-else>
          <label>
            <span>{{ t('name') }}</span>
            <el-input
              :model-value="row.data.name || ''"
              placeholder="http_api / search_api / weather_api"
              @update:model-value="updateField(index, 'name', $event)"
            />
          </label>
        </template>

        <label>
          <span>{{ t('description') }}</span>
          <el-input
            :model-value="row.data.description || ''"
            type="textarea"
            :autosize="{ minRows: 3, maxRows: 8 }"
            :placeholder="t('toolDescriptionPlaceholder')"
            @update:model-value="updateField(index, 'description', $event)"
          />
        </label>

        <template v-if="row.data.binding === 'api'">
          <div class="subsection-title">{{ t('apiConfiguration') }}</div>
          <label>
            <span>{{ t('method') }}</span>
            <el-select :model-value="row.data.config.method" @update:model-value="updateConfigField(index, 'method', $event)">
              <el-option v-for="method in httpMethods" :key="method" :value="method" :label="method" />
            </el-select>
          </label>
          <label>
            <span>{{ t('url') }}</span>
            <el-input
              :model-value="row.data.config.url || ''"
              placeholder="https://example.com/endpoint"
              @update:model-value="updateConfigField(index, 'url', $event)"
            />
          </label>
          <label>
            <span>{{ t('responseMode') }}</span>
            <el-select :model-value="row.data.config.response" @update:model-value="updateConfigField(index, 'response', $event)">
              <el-option v-for="mode in responseModes" :key="mode" :value="mode" :label="mode" />
            </el-select>
          </label>
          <label>
            <span>{{ t('timeoutSeconds') }}</span>
            <el-input-number
              class="number-input"
              :min="1"
              :step="1"
              :model-value="row.data.config.timeout"
              @update:model-value="updateConfigField(index, 'timeout', Number($event || 20))"
            />
          </label>
          <label>
            <span>{{ t('headersJson') }}</span>
            <el-input
              :model-value="JSON.stringify(row.data.config.headers || {}, null, 2)"
              type="textarea"
              :autosize="{ minRows: 4, maxRows: 10 }"
              placeholder='{"Authorization":"Bearer YOUR_KEY"}'
              @update:model-value="updateJsonConfigField(index, 'headers', $event)"
            />
          </label>
          <label>
            <span>{{ t('queryParamsJson') }}</span>
            <el-input
              :model-value="JSON.stringify(row.data.config.params || {}, null, 2)"
              type="textarea"
              :autosize="{ minRows: 4, maxRows: 10 }"
              placeholder='{"q":"{query}"}'
              @update:model-value="updateJsonConfigField(index, 'params', $event)"
            />
          </label>
          <label>
            <span>{{ t('bodyJson') }}</span>
            <el-input
              :model-value="JSON.stringify(row.data.config.body ?? {}, null, 2)"
              type="textarea"
              :autosize="{ minRows: 4, maxRows: 10 }"
              placeholder='{"query":"{query}"}'
              @update:model-value="updateJsonConfigField(index, 'body', $event)"
            />
          </label>
          <div class="field-help">
            {{ t('apiPlaceholderHelp') }}
          </div>
        </template>

        <template v-else-if="row.data.binding === 'mcp'">
          <div class="field-help">
            {{ t('mcpToolHelp') }}
          </div>
        </template>

        <el-button class="mini-button align-end" @click="removeTool(index)">{{ t('remove') }}</el-button>
      </div>
    </div>

    <el-button type="primary" class="mini-button" @click="addTool()">{{ t('addTool') }}</el-button>
  </div>
</template>
