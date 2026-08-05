<script setup>
import { computed } from 'vue';
import { useI18n } from '../composables/useI18n';

const props = defineProps({
  value: { type: Object, default: () => ({ style: '', rules: [] }) },
  help: { type: String, default: '' },
  ruleSuggestions: { type: Array, default: () => [] }
});

const emit = defineEmits(['update:value']);
const { t } = useI18n();

const rulesText = computed(() => (props.value?.rules || []).join('\n'));

function splitRules(value) {
  return String(value || '')
    .split(/\r?\n/)
    .map((line) => line.replace(/^[-*]\s*/, '').replace(/^\d+[.)]\s*/, '').trim())
    .filter(Boolean);
}

function updateRules(value) {
  emit('update:value', {
    ...props.value,
    style: '',
    rules: splitRules(value)
  });
}

function appendRule(rule) {
  const next = [...(props.value?.rules || [])];
  if (!next.includes(rule)) next.push(rule);
  emit('update:value', {
    ...props.value,
    style: '',
    rules: next
  });
}
</script>

<template>
  <div class="structured-editor">
    <div v-if="help" class="field-help">{{ help }}</div>
    <label>
      <span>{{ t('globalRules') }}</span>
      <el-input
        :model-value="rulesText"
        type="textarea"
        :autosize="{ minRows: 8, maxRows: 16 }"
        :placeholder="t('globalRulesPlaceholder')"
        @update:model-value="updateRules"
      />
    </label>
    <div v-if="ruleSuggestions.length" class="suggestion-strip">
      <div class="suggestion-title">{{ t('suggestedRules') }}</div>
      <div class="suggestion-list">
        <el-button
          v-for="rule in ruleSuggestions"
          :key="rule"
          class="suggestion-chip"
          text
          @click="appendRule(rule)"
        >
          <span>{{ rule }}</span>
        </el-button>
      </div>
    </div>
  </div>
</template>
