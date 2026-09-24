function asNumber(value) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function isRecord(value) {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

function compactText(value, max = 160) {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim();
  if (!text) return '';
  if (text.length <= max) return text;
  return `${text.slice(0, Math.max(0, max - 3))}...`;
}

function formatIso(ts) {
  const n = asNumber(ts);
  if (n === null) return 'n/a';
  try {
    return new Date(n).toISOString();
  } catch {
    return 'n/a';
  }
}

function formatTokenParts(total, prompt, completion) {
  if (total === null && prompt === null && completion === null) return 'n/a';
  if (total !== null && prompt !== null && completion !== null) {
    return `${Math.round(total)} (p=${Math.round(prompt)}, c=${Math.round(completion)})`;
  }
  if (total !== null) return String(Math.round(total));
  const parts = [];
  if (prompt !== null) parts.push(`p=${Math.round(prompt)}`);
  if (completion !== null) parts.push(`c=${Math.round(completion)}`);
  return parts.join(', ') || 'n/a';
}

/**
 * @param {unknown} value
 * @returns {{ total: number | null; prompt: number | null; completion: number | null } | null}
 */
export function extractTokenUsage(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const v = /** @type {Record<string, unknown>} */ (value);

  const tokenUsageObj = isRecord(v.token_usage) ? v.token_usage : isRecord(v.tokenUsage) ? v.tokenUsage : null;
  if (tokenUsageObj) {
    const total = asNumber(tokenUsageObj.total_tokens ?? tokenUsageObj.totalTokens ?? tokenUsageObj.total);
    const prompt = asNumber(tokenUsageObj.prompt_tokens ?? tokenUsageObj.promptTokens ?? tokenUsageObj.prompt);
    const completion = asNumber(
      tokenUsageObj.completion_tokens ?? tokenUsageObj.completionTokens ?? tokenUsageObj.completion
    );
    if (total !== null || prompt !== null || completion !== null) {
      return { total, prompt, completion };
    }
  }

  for (const key of ['token_usage', 'tokenUsage', 'tokens', 'total_tokens', 'totalTokens']) {
    const direct = asNumber(v[key]);
    if (direct !== null) return { total: direct, prompt: null, completion: null };
  }

  const usage = isRecord(v.usage) ? v.usage : null;
  if (usage) {
    const total = asNumber(usage.total_tokens ?? usage.totalTokens ?? usage.total);
    const prompt = asNumber(usage.prompt_tokens ?? usage.promptTokens ?? usage.prompt);
    const completion = asNumber(usage.completion_tokens ?? usage.completionTokens ?? usage.completion);
    if (total !== null || prompt !== null || completion !== null) {
      return { total, prompt, completion };
    }
  }

  const metrics = isRecord(v.metrics) ? v.metrics : null;
  if (metrics) {
    const nested = extractTokenUsage(metrics);
    if (nested) return nested;
  }

  return null;
}

/**
 * @param {number | null | undefined} ms
 * @returns {string}
 */
export function formatDurationMs(ms) {
  const value = asNumber(ms);
  if (value === null) return 'n/a';
  if (value < 1000) return `${Math.round(value)} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(2)} s`;
  if (value < 3_600_000) return `${(value / 60_000).toFixed(2)} min`;
  return `${(value / 3_600_000).toFixed(2)} h`;
}

/**
 * @param {{
 *   session?: any;
 *   archivedSession?: any;
 *   graph?: any;
 *   exec?: any;
 *   humanRequests?: any[];
 * }} args
 */
export function buildSessionSummary(args) {
  const session = args?.session ?? null;
  const archivedSession = args?.archivedSession ?? null;
  const graph = args?.graph ?? null;
  const exec = args?.exec ?? { runningNodes: [], nodeHistory: {} };
  const humanRequests = Array.isArray(args?.humanRequests) ? args.humanRequests : [];
  const nodeHistory = isRecord(exec?.nodeHistory) ? exec.nodeHistory : {};

  let totalDurationMs = 0;
  let totalTokens = 0;
  let promptTokens = 0;
  let completionTokens = 0;
  let hasTokenData = false;
  let hasPromptData = false;
  let hasCompletionData = false;
  let completedRunCount = 0;
  let okRunCount = 0;
  let errorRunCount = 0;
  let runningRunCount = 0;
  let executedNodeCount = 0;
  let firstStartedAt = null;
  let latestActivityAt = null;

  /** @type {Map<string, { status: 'ok' | 'error'; sortTs: number }>} */
  const latestFinishedByNode = new Map();

  for (const [nodeId, historyValue] of Object.entries(nodeHistory)) {
    const history = Array.isArray(historyValue) ? historyValue : [];
    if (history.length > 0) executedNodeCount += 1;

    for (const run of history) {
      if (!run || typeof run !== 'object') continue;
      const startedAt = asNumber(run.startedAt);
      const endedAt = asNumber(run.endedAt);
      const status = run.status === 'error' ? 'error' : run.status === 'ok' ? 'ok' : 'running';

      if (startedAt !== null) {
        firstStartedAt = firstStartedAt === null ? startedAt : Math.min(firstStartedAt, startedAt);
        latestActivityAt = latestActivityAt === null ? startedAt : Math.max(latestActivityAt, startedAt);
      }
      if (endedAt !== null) {
        latestActivityAt = latestActivityAt === null ? endedAt : Math.max(latestActivityAt, endedAt);
      }

      if (status === 'running' && endedAt === null) {
        runningRunCount += 1;
      } else {
        completedRunCount += 1;
        if (status === 'ok') okRunCount += 1;
        if (status === 'error') errorRunCount += 1;
      }

      if (startedAt !== null && endedAt !== null) {
        totalDurationMs += Math.max(0, endedAt - startedAt);
      }

      const tokenUsage = extractTokenUsage(run.metrics) || extractTokenUsage(run.outputs) || extractTokenUsage(run.inputs);
      if (tokenUsage) {
        const total = asNumber(tokenUsage.total);
        const prompt = asNumber(tokenUsage.prompt);
        const completion = asNumber(tokenUsage.completion);
        if (total !== null) {
          totalTokens += total;
          hasTokenData = true;
        }
        if (prompt !== null) {
          promptTokens += prompt;
          hasPromptData = true;
          hasTokenData = true;
        }
        if (completion !== null) {
          completionTokens += completion;
          hasCompletionData = true;
          hasTokenData = true;
        }
      }

      if (status === 'ok' || status === 'error') {
        const sortTs = endedAt ?? startedAt ?? 0;
        const prev = latestFinishedByNode.get(nodeId);
        if (!prev || sortTs >= prev.sortTs) {
          latestFinishedByNode.set(nodeId, { status, sortTs });
        }
      }
    }
  }

  let okNodeCount = 0;
  let errorNodeCount = 0;
  for (const item of latestFinishedByNode.values()) {
    if (item.status === 'ok') okNodeCount += 1;
    if (item.status === 'error') errorNodeCount += 1;
  }

  const sessionStart = firstStartedAt ?? asNumber(session?.connectedAt) ?? null;
  const archivedEndedAt = asNumber(archivedSession?.endedAt);
  const liveLastSeenAt = asNumber(session?.lastSeenAt);
  const sessionEnd =
    archivedEndedAt ??
    (latestActivityAt !== null || liveLastSeenAt !== null
      ? Math.max(latestActivityAt ?? 0, liveLastSeenAt ?? 0)
      : asNumber(session?.connectedAt) ?? null);
  const elapsedMs = sessionStart !== null && sessionEnd !== null ? Math.max(0, sessionEnd - sessionStart) : null;

  const pendingHumanRequests = humanRequests.filter((item) => item && !item.resolved).length;
  const resolvedHumanRequests = humanRequests.filter((item) => item && item.resolved).length;
  const totalHumanRequests = humanRequests.length;

  return {
    totalTokens: hasTokenData ? totalTokens : null,
    promptTokens: hasPromptData ? promptTokens : null,
    completionTokens: hasCompletionData ? completionTokens : null,
    totalDurationMs,
    elapsedMs,
    okNodeCount,
    errorNodeCount,
    okRunCount,
    errorRunCount,
    completedRunCount,
    runningRunCount,
    executedNodeCount,
    totalHumanRequests,
    pendingHumanRequests,
    resolvedHumanRequests,
    graphNodeCount: Array.isArray(graph?.nodes) ? graph.nodes.length : null,
    graphEdgeCount: Array.isArray(graph?.edges) ? graph.edges.length : null
  };
}

/**
 * @param {{
 *   session?: any;
 *   archivedSession?: any;
 *   graph?: any;
 *   exec?: any;
 *   debugState?: any;
 *   logs?: any[];
 *   systemEvents?: any[];
 *   flows?: any[];
 *   humanRequests?: any[];
 *   humanChats?: any[];
 * }} args
 */
export function buildSessionExportData(args) {
  const summary = buildSessionSummary(args);
  const session = args?.session ?? null;
  const archivedSession = args?.archivedSession ?? null;
  return {
    exportedAt: new Date().toISOString(),
    session: {
      id: session?.id ?? archivedSession?.id ?? '',
      graphName: session?.graphName ?? archivedSession?.graphName ?? null,
      mode: session?.mode ?? archivedSession?.mode ?? 'unknown',
      pid: session?.pid ?? archivedSession?.pid ?? null,
      connectedAt: session?.connectedAt ?? archivedSession?.connectedAt ?? null,
      lastSeenAt: session?.lastSeenAt ?? archivedSession?.lastSeenAt ?? null,
      endedAt: archivedSession?.endedAt ?? null,
      subscribed: !!session?.subscribed,
      exited: !!archivedSession && !session
    },
    summary,
    graph: args?.graph ?? null,
    execution: args?.exec ?? { runningNodes: [], nodeHistory: {} },
    debug: args?.debugState ?? null,
    human: {
      requests: Array.isArray(args?.humanRequests) ? args.humanRequests : [],
      chats: Array.isArray(args?.humanChats) ? args.humanChats : []
    },
    logs: {
      program: Array.isArray(args?.logs) ? args.logs : [],
      system: Array.isArray(args?.systemEvents) ? args.systemEvents : []
    },
    flows: Array.isArray(args?.flows) ? args.flows : []
  };
}

/**
 * @param {ReturnType<typeof buildSessionExportData>} payload
 * @returns {string}
 */
export function formatSessionExportMarkdown(payload) {
  const summary = payload.summary;
  const errors = [];
  const nodeHistory = isRecord(payload.execution?.nodeHistory) ? payload.execution.nodeHistory : {};
  for (const [nodeId, historyValue] of Object.entries(nodeHistory)) {
    const history = Array.isArray(historyValue) ? historyValue : [];
    for (const run of history) {
      if (!run || typeof run !== 'object') continue;
      if (run.status !== 'error' && !run.error) continue;
      errors.push({
        nodeId,
        runId: typeof run.runId === 'string' ? run.runId : '',
        startedAt: formatIso(run.startedAt),
        error: compactText(run.error || 'Unknown error', 300)
      });
    }
  }

  const lines = [];
  lines.push('# MASFactory Session Export');
  lines.push('');
  lines.push(`- Graph: ${payload.session.graphName || '(unknown graph)'}`);
  lines.push(`- Session: \`${payload.session.id || 'n/a'}\``);
  lines.push(`- Mode: ${payload.session.mode || 'unknown'}`);
  lines.push(`- PID: ${payload.session.pid ?? 'n/a'}`);
  lines.push(`- Exported At: ${payload.exportedAt}`);
  lines.push(`- Connected At: ${formatIso(payload.session.connectedAt)}`);
  lines.push(`- Last Seen At: ${formatIso(payload.session.lastSeenAt)}`);
  if (payload.session.endedAt) lines.push(`- Ended At: ${formatIso(payload.session.endedAt)}`);
  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push('| Metric | Value |');
  lines.push('| --- | --- |');
  lines.push(`| Tokens | ${formatTokenParts(summary.totalTokens, summary.promptTokens, summary.completionTokens)} |`);
  lines.push(`| Total Duration | ${formatDurationMs(summary.totalDurationMs)} |`);
  lines.push(`| Elapsed | ${formatDurationMs(summary.elapsedMs)} |`);
  lines.push(`| OK Nodes | ${summary.okNodeCount} |`);
  lines.push(`| Error Nodes | ${summary.errorNodeCount} |`);
  lines.push(`| OK Runs | ${summary.okRunCount} |`);
  lines.push(`| Error Runs | ${summary.errorRunCount} |`);
  lines.push(`| Running Runs | ${summary.runningRunCount} |`);
  lines.push(`| Human Requests | ${summary.totalHumanRequests} |`);
  lines.push(`| Pending Human Requests | ${summary.pendingHumanRequests} |`);
  lines.push(`| Executed Nodes | ${summary.executedNodeCount} |`);
  lines.push(`| Graph Nodes | ${summary.graphNodeCount ?? 'n/a'} |`);
  lines.push(`| Graph Edges | ${summary.graphEdgeCount ?? 'n/a'} |`);

  const requests = Array.isArray(payload.human?.requests) ? payload.human.requests : [];
  if (requests.length > 0) {
    lines.push('');
    lines.push('## Human Requests');
    lines.push('');
    for (const req of requests) {
      if (!req || typeof req !== 'object') continue;
      const status = req.resolved ? 'resolved' : 'pending';
      const node = req.node ? ` @${req.node}` : '';
      lines.push(`- [${status}] \`${req.requestId || 'n/a'}\`${node}: ${compactText(req.prompt || '', 220)}`);
    }
  }

  if (errors.length > 0) {
    lines.push('');
    lines.push('## Error Runs');
    lines.push('');
    for (const err of errors) {
      lines.push(`- \`${err.nodeId}\` (${err.runId || 'n/a'}, ${err.startedAt}): ${err.error}`);
    }
  }

  const programLogs = Array.isArray(payload.logs?.program) ? payload.logs.program.slice(-20) : [];
  if (programLogs.length > 0) {
    lines.push('');
    lines.push('## Recent Program Logs');
    lines.push('');
    for (const item of programLogs) {
      lines.push(`- [${formatIso(item.ts)}] ${String(item.level || 'info').toUpperCase()}: ${compactText(item.message || '', 240)}`);
    }
  }

  const systemEvents = Array.isArray(payload.logs?.system) ? payload.logs.system.slice(-20) : [];
  if (systemEvents.length > 0) {
    lines.push('');
    lines.push('## Recent System Events');
    lines.push('');
    for (const item of systemEvents) {
      if (item?.kind === 'trace') {
        lines.push(`- [${formatIso(item.ts)}] TRACE ${item.dir || '?'} ${item.messageType || 'unknown'}`);
        continue;
      }
      lines.push(`- [${formatIso(item?.ts)}] ${String(item?.level || 'info').toUpperCase()}: ${compactText(item?.message || '', 240)}`);
    }
  }

  return `${lines.join('\n')}\n`;
}

/**
 * @param {string | null | undefined} graphName
 * @param {string | null | undefined} sessionId
 * @param {'json' | 'markdown'} format
 * @returns {string}
 */
export function makeSessionExportFileName(graphName, sessionId, format) {
  const base = String(graphName || 'session').trim() || 'session';
  const sid = String(sessionId || 'session').trim() || 'session';
  const safe = `${base}-${sid}`
    .replace(/[^A-Za-z0-9._-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 120);
  return `${safe || 'session-export'}.${format === 'markdown' ? 'md' : 'json'}`;
}
