import type { GraphData } from '../types/graph';
import type { ExecutionState } from '../types/runtimeExec';
import type {
  ArchivedSession,
  DebugSessionState,
  HumanChatMessage,
  HumanInteractionRequest,
  RuntimeFlowEntry,
  RuntimeLogEntry,
  RuntimeTraceEntry,
  VisualizerSession
} from '../stores/runtimeTypes';

export type SessionSystemEvent =
  | { kind: 'log'; ts: number; level: 'info' | 'warn' | 'error'; message: string }
  | { kind: 'trace'; ts: number; dir: 'in' | 'out'; messageType: string; payload?: unknown };

export interface SessionSummary {
  totalTokens: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  totalDurationMs: number;
  elapsedMs: number | null;
  okNodeCount: number;
  errorNodeCount: number;
  okRunCount: number;
  errorRunCount: number;
  completedRunCount: number;
  runningRunCount: number;
  executedNodeCount: number;
  totalHumanRequests: number;
  pendingHumanRequests: number;
  resolvedHumanRequests: number;
  graphNodeCount: number | null;
  graphEdgeCount: number | null;
}

export interface SessionExportData {
  exportedAt: string;
  session: {
    id: string;
    graphName: string | null;
    mode: string;
    pid: number | null;
    connectedAt: number | null;
    lastSeenAt: number | null;
    endedAt: number | null;
    subscribed: boolean;
    exited: boolean;
  };
  summary: SessionSummary;
  graph: GraphData | null;
  execution: ExecutionState;
  debug: DebugSessionState | null;
  human: {
    requests: HumanInteractionRequest[];
    chats: HumanChatMessage[];
  };
  logs: {
    program: RuntimeLogEntry[];
    system: SessionSystemEvent[];
  };
  flows: RuntimeFlowEntry[];
}

export function extractTokenUsage(
  value: unknown
): { total: number | null; prompt: number | null; completion: number | null } | null;

export function formatDurationMs(ms: number | null | undefined): string;

export function buildSessionSummary(args: {
  session?: VisualizerSession | null;
  archivedSession?: ArchivedSession | null;
  graph?: GraphData | null;
  exec?: ExecutionState | null;
  humanRequests?: HumanInteractionRequest[] | null;
}): SessionSummary;

export function buildSessionExportData(args: {
  session?: VisualizerSession | null;
  archivedSession?: ArchivedSession | null;
  graph?: GraphData | null;
  exec?: ExecutionState | null;
  debugState?: DebugSessionState | null;
  logs?: RuntimeLogEntry[] | null;
  systemEvents?: SessionSystemEvent[] | null;
  flows?: RuntimeFlowEntry[] | null;
  humanRequests?: HumanInteractionRequest[] | null;
  humanChats?: HumanChatMessage[] | null;
}): SessionExportData;

export function formatSessionExportMarkdown(payload: SessionExportData): string;

export function makeSessionExportFileName(
  graphName: string | null | undefined,
  sessionId: string | null | undefined,
  format: 'json' | 'markdown'
): string;
