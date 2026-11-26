export interface PlaylistStat {
  id: string;
  name: string;
  url: string;
  total_count: number;
  indexed_count: number;
  new_count?: number;
  no_transcript_count?: number;
  new_videos?: Array<{ video_id: string; title: string }>;
}

export interface SyncEvent {
  playlist?: string;
  playlist_id?: string;
  current?: number;
  total?: number;
  video_id?: string;
  video_title?: string;
  status: string;
  message?: string;
  chunk_count?: number;
}

export interface SourceItem {
  video_id: string;
  title: string;
  upload_date: string;
  start_sec: number;
  end_sec: number;
  timestamp_str: string;
  url: string;
  snippet: string;
}

export interface ChatResponse {
  answer: string;
  mode: 'lookup' | 'opinion' | 'evolution' | 'freeform' | 'abhijit_ai';
  sources: SourceItem[];
}

export interface RateLimitStatus {
  is_rate_limited: boolean;
  message?: string | null;
  timestamp?: string | null;
}

export async function fetchSyncStatus(): Promise<RateLimitStatus> {
  const res = await fetch('/api/sync/status');
  if (!res.ok) throw new Error(`Failed to fetch sync status: ${res.statusText}`);
  return res.json();
}

export async function clearRateLimitApi(): Promise<void> {
  await fetch('/api/sync/clear-rate-limit', { method: 'POST' });
}

export interface AppConfig {
  read_only_mode: boolean;
  has_server_key: boolean;
}

export async function fetchAppConfig(): Promise<AppConfig> {
  const res = await fetch('/api/config');
  if (!res.ok) throw new Error(`Failed to fetch config: ${res.statusText}`);
  return res.json();
}

export async function fetchPlaylists(checkRemote: boolean = false, adminToken?: string): Promise<PlaylistStat[]> {
  const url = checkRemote ? '/api/playlists?check_remote=true' : '/api/playlists';
  const headers: Record<string, string> = {};
  if (adminToken) headers['X-Admin-Token'] = adminToken;
  const res = await fetch(url, { headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Failed to fetch playlists: ${res.statusText}`);
  }
  return res.json();
}

export async function syncPlaylistsStream(
  playlistIds?: string[],
  maxVideosPerPlaylist?: number,
  onEvent?: (event: SyncEvent) => void,
  adminToken?: string,
): Promise<void> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (adminToken) headers['X-Admin-Token'] = adminToken;
  const res = await fetch('/api/sync', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      playlist_ids: playlistIds,
      max_videos_per_playlist: maxVideosPerPlaylist,
    }),
  });

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Sync request failed: ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data: ')) {
        const jsonStr = trimmed.slice(6);
        try {
          const parsed = JSON.parse(jsonStr) as SyncEvent;
          if (onEvent) onEvent(parsed);
        } catch {
          // Ignore JSON parse errors for incomplete chunks
        }
      }
    }
  }
}

export interface ChatStreamEvent {
  type: 'sources' | 'delta' | 'done' | 'error';
  mode?: 'lookup' | 'opinion' | 'evolution' | 'freeform' | 'abhijit_ai';
  sources?: SourceItem[];
  text?: string;
  message?: string;
}

export async function sendChatMessageStream(
  question: string,
  playlistId: string | null = null,
  mode: 'auto' | 'lookup' | 'opinion' | 'evolution' | 'freeform' | 'abhijit_ai' = 'auto',
  provider: string = 'ollama',
  apiKey: string = 'ollama',
  onEvent?: (event: ChatStreamEvent) => void,
): Promise<void> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Provider': provider,
  };
  if (apiKey) {
    headers['X-Api-Key'] = apiKey;
  }
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      question,
      playlist_id: playlistId || null,
      mode,
      stream: true,
    }),
  });

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to process chat message');
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data: ')) {
        const jsonStr = trimmed.slice(6);
        try {
          const parsed = JSON.parse(jsonStr) as ChatStreamEvent;
          if (parsed.type === 'error') {
            throw new Error(parsed.message || 'Error from streaming endpoint');
          }
          if (onEvent) onEvent(parsed);
        } catch (e: any) {
          if (e.message && e.message !== 'Unexpected end of JSON input') {
            throw e;
          }
        }
      }
    }
  }
}

export async function sendChatMessage(
  question: string,
  playlistId: string | null = null,
  mode: 'auto' | 'lookup' | 'opinion' | 'evolution' | 'freeform' | 'abhijit_ai' = 'auto',
  provider: string = 'ollama',
  apiKey: string = 'ollama',
): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'X-Provider': provider,
  };
  if (apiKey) {
    headers['X-Api-Key'] = apiKey;
  }
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      question,
      playlist_id: playlistId || null,
      mode,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to process chat message');
  }

  return res.json();
}
