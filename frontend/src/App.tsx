import React, { useEffect, useState } from 'react';
import {
  ExternalLink,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp,
  History,
  CheckCircle2,
  AlertCircle,
  AlertTriangle,
  X,
  Lock,
  MessageSquare,
  Bot,
  Terminal,
  Cpu,
  Radio,
  CornerDownLeft,
  Zap,
  Database,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import {
  fetchAppConfig,
  fetchPlaylists,
  fetchSyncStatus,
  clearRateLimitApi,
  syncPlaylistsStream,
  sendChatMessageStream,
  PlaylistStat,
  ChatResponse,
  SyncEvent,
  AppConfig,
} from './api';

function formatUploadDate(dateStr?: string): string {
  if (!dateStr || dateStr === 'N/A') return 'N/A';
  const clean = dateStr.trim();
  if (/^\d{8}$/.test(clean)) {
    return `${clean.slice(0, 4)}-${clean.slice(4, 6)}-${clean.slice(6, 8)}`;
  }
  return clean;
}

export default function App() {
  const [appConfig, setAppConfig] = useState<AppConfig>({ read_only_mode: false, has_server_key: true });
  const provider = 'ollama';
  const apiKey = 'ollama';

  const [playlists, setPlaylists] = useState<PlaylistStat[]>([]);
  const [selectedPlaylist, setSelectedPlaylist] = useState<string>('all');
  const [mode, setMode] = useState<'abhijit_ai' | 'lookup' | 'opinion' | 'evolution' | 'freeform'>('abhijit_ai');
  const [question, setQuestion] = useState('');
  const [loadingChat, setLoadingChat] = useState(false);
  const [chatResponse, setChatResponse] = useState<ChatResponse | null>(null);
  const [chatError, setChatError] = useState<string | null>(null);

  // Sync state
  const [checkingRemote, setCheckingRemote] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncProgress, setSyncProgress] = useState<{
    current: number;
    total: number;
    message: string;
    status?: string;
  } | null>(null);
  const [rateLimitInfo, setRateLimitInfo] = useState<{
    message: string;
    timestamp?: string;
    playlist?: string;
    playlistId?: string;
    videoId?: string;
  } | null>(null);
  const [batchSize, setBatchSize] = useState<number | 'all'>(28);
  const [showSources, setShowSources] = useState(true);

  // Load config, playlists and sync status on mount
  useEffect(() => {
    loadAppConfig();
    loadPlaylists(false);
    checkRateLimitStatus();
  }, []);

  async function loadAppConfig() {
    try {
      const cfg = await fetchAppConfig();
      setAppConfig(cfg);
    } catch (err) {
      console.warn('Failed to fetch app config:', err);
    }
  }

  async function checkRateLimitStatus() {
    try {
      const status = await fetchSyncStatus();
      if (status.is_rate_limited && status.message) {
        setRateLimitInfo({
          message: status.message,
          timestamp: status.timestamp ? new Date(status.timestamp).toLocaleTimeString() : undefined,
        });
      }
    } catch (err) {
      console.warn('Could not check rate limit status:', err);
    }
  }

  async function handleDismissRateLimit() {
    setRateLimitInfo(null);
    try {
      await clearRateLimitApi();
    } catch (err) {
      console.warn('Failed to clear rate limit on backend:', err);
    }
  }

  async function loadPlaylists(checkRemote: boolean) {
    try {
      if (checkRemote) setCheckingRemote(true);
      const data = await fetchPlaylists(checkRemote);
      setPlaylists(data);
      if (checkRemote) {
        await checkRateLimitStatus();
      }
    } catch (err: any) {
      console.error(err);
      if (err.message?.includes('429') || err.message?.toLowerCase().includes('rate limit')) {
        setRateLimitInfo({
          message: 'Rate limit exceeded while checking playlist updates (HTTP 429). Sync is paused to protect IP.',
          timestamp: new Date().toLocaleTimeString(),
        });
      }
    } finally {
      if (checkRemote) setCheckingRemote(false);
    }
  }

  async function handleSync(playlistIds?: string[], limit?: number) {
    try {
      setSyncing(true);
      let encounteredRateLimit = false;
      const effectiveLimit = limit !== undefined ? limit : (batchSize === 'all' ? undefined : batchSize);
      setSyncProgress({ current: 0, total: 1, message: 'Initiating neural ingest pipeline...', status: 'processing' });
      await syncPlaylistsStream(playlistIds, effectiveLimit, (event: SyncEvent) => {
        if (event.status === 'rate_limited') {
          encounteredRateLimit = true;
          const msg = event.message || 'YouTube rate-limited transcript requests from this IP (429). Pausing sync to protect IP.';
          setRateLimitInfo({
            message: msg,
            timestamp: new Date().toLocaleTimeString(),
            playlist: event.playlist,
            playlistId: event.playlist_id,
            videoId: event.video_id,
          });
          setSyncProgress({
            current: event.current || 0,
            total: event.total || 1,
            message: `⚠️ [ERR_429] Rate limit throttled: ${msg}`,
            status: 'rate_limited',
          });
          return;
        }

        if (event.status === 'done') {
          setRateLimitInfo(null);
        }

        if (event.total && event.total > 0) {
          setSyncProgress({
            current: event.current || 0,
            total: event.total,
            message: event.message || `Processing ${event.video_title || ''}`,
            status: event.status,
          });
        } else if (event.message) {
          setSyncProgress((prev) => ({
            current: prev?.current || 0,
            total: prev?.total || 1,
            message: event.message || '',
            status: event.status || prev?.status,
          }));
        }
      });
      await loadPlaylists(false);
      if (!encounteredRateLimit) {
        setRateLimitInfo(null);
      }
    } catch (err: any) {
      if (err.message?.includes('429') || err.message?.toLowerCase().includes('rate limit')) {
        setRateLimitInfo({
          message: `Sync throttled: ${err.message}`,
          timestamp: new Date().toLocaleTimeString(),
        });
      } else {
        alert(`Sync failed: ${err.message}`);
      }
    } finally {
      setSyncing(false);
      setSyncProgress(null);
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim() || loadingChat) return;

    setLoadingChat(true);
    setChatError(null);
    setChatResponse(null);
    try {
      const plId = selectedPlaylist === 'all' ? null : selectedPlaylist;
      await sendChatMessageStream(
        question,
        plId,
        mode,
        provider,
        apiKey,
        (event) => {
          if (event.type === 'sources') {
            setChatResponse({
              answer: '',
              mode: event.mode || mode,
              sources: event.sources || [],
            });
          } else if (event.type === 'delta' && event.text) {
            setChatResponse((prev) => {
              if (!prev) {
                return {
                  answer: event.text || '',
                  mode: event.mode || mode,
                  sources: [],
                };
              }
              return {
                ...prev,
                answer: prev.answer + event.text,
              };
            });
          }
        },
      );
    } catch (err: any) {
      setChatError(err.message || 'Neural synthesis error');
    } finally {
      setLoadingChat(false);
    }
  }

  const modeLabels: Record<string, { label: string; desc: string; icon: any; color: string; badge: string }> = {
    abhijit_ai: {
      label: 'Abhijit AI',
      desc: "Synthesize response in Abhijit Chavda's voice & convictions (#AskAbhijit)",
      icon: Bot,
      color: 'text-red-400 border-red-500/50 bg-red-950/30',
      badge: 'border-red-500/50 text-red-400 bg-red-950/40',
    },
    lookup: {
      label: 'Find Timestamp',
      desc: 'Locate exact quotes & video moments with clickable timestamps',
      icon: Search,
      color: 'text-white border-zinc-700 bg-zinc-800',
      badge: 'border-zinc-700 text-zinc-200 bg-zinc-900',
    },
    opinion: {
      label: 'Overall Opinion',
      desc: 'Synthesize consistent viewpoint across distinct videos',
      icon: Layers,
      color: 'text-zinc-200 border-zinc-700 bg-zinc-800',
      badge: 'border-zinc-700 text-zinc-200 bg-zinc-900',
    },
    evolution: {
      label: 'Opinion Over Time',
      desc: 'Chronological stance analysis and perspective evolution',
      icon: History,
      color: 'text-amber-400 border-amber-500/40 bg-amber-950/30',
      badge: 'border-amber-500/50 text-amber-400 bg-amber-950/40',
    },
    freeform: {
      label: 'Freeform',
      desc: 'Open conversational answer without video constraints (#AskAbhijit 9000)',
      icon: MessageSquare,
      color: 'text-zinc-200 border-zinc-700 bg-zinc-800',
      badge: 'border-zinc-700 text-zinc-200 bg-zinc-900',
    },
  };

  const totalIndexedVideos = playlists.reduce((acc, p) => acc + (p.indexed_count || 0), 0);

  return (
    <div className="flex h-screen w-screen bg-black text-zinc-100 overflow-hidden font-sans selection:bg-red-950 selection:text-red-200">
      {/* LEFT SIDEBAR: Telemetry & Ingestion Matrix */}
      <aside className="w-80 md:w-[370px] border-r border-zinc-800/80 bg-[#09090b] flex flex-col shrink-0 relative z-20">
        {/* Top Header / Branding */}
        <div className="p-4 border-b border-zinc-800/80 bg-zinc-950/60">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2.5">
              <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-red-600/10 border border-red-500/30 shadow-[0_0_12px_rgba(255,0,51,0.2)]">
                <Cpu className="w-4 h-4 text-red-500" />
                <div className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-red-500 ring-2 ring-black animate-ping" />
              </div>
              <div>
                <div className="flex items-center gap-1.5">
                  <h1 className="font-tech text-base tracking-widest text-white font-bold">
                    yt-rag
                  </h1>
                  <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-zinc-900 text-zinc-400 border border-zinc-800 uppercase tracking-tight">
                    RAG Engine
                  </span>
                </div>
                <p className="text-[10px] font-mono text-zinc-400 tracking-wider">
                  Abhijit Chavda Archive
                </p>
              </div>
            </div>

            {/* Read-Only or Online Badge */}
            {appConfig.read_only_mode ? (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-zinc-900 text-zinc-400 border border-zinc-800 shadow-sm shrink-0 flex items-center gap-1">
                <Lock className="w-2.5 h-2.5" />
                READ-ONLY
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-500/40 flex items-center gap-1 shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                READY
              </span>
            )}
          </div>

          <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400 pt-1 border-t border-zinc-800/50">
            <span>INDEXED: <strong className="text-white">{totalIndexedVideos}</strong> VIDEOS</span>
            <span>PLAYLISTS: <strong className="text-zinc-200">{playlists.length}</strong></span>
          </div>
        </div>

        {/* Local Ollama Configuration Card */}
        <div className="p-3.5 border-b border-zinc-800/80 bg-zinc-950/30 space-y-2.5">
          <div className="flex items-center justify-between">
            <label className="text-[10px] font-mono uppercase tracking-wider text-zinc-300 flex items-center gap-1.5">
              <Bot className="w-3 h-3 text-red-500" />
              Local Model (Ollama)
            </label>
            <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
              ONLINE
            </span>
          </div>

          <div className="p-2.5 rounded-lg bg-zinc-900/70 border border-zinc-800 space-y-1.5 text-xs font-mono">
            <div className="flex items-center justify-between">
              <span className="text-zinc-400 text-[11px]">Chat LLM:</span>
              <span className="text-zinc-200 text-[11px]">llama3.2:3b</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-zinc-400 text-[11px]">Embeddings:</span>
              <span className="text-zinc-200 text-[11px]">nomic-embed-text</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-zinc-400 text-[11px]">Endpoint:</span>
              <span className="text-zinc-400 text-[10px]">http://localhost:11434</span>
            </div>
          </div>
        </div>

        {/* Sync Controls */}
        <div className="p-4 border-b border-zinc-800/80 space-y-3">
          {appConfig.read_only_mode ? (
            <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800 text-zinc-400 text-xs flex items-center gap-2 font-mono">
              <Lock className="w-4 h-4 text-zinc-400 shrink-0" />
              <span className="text-[11px] leading-tight">
                Hosted Snapshot: Playlist sync is read-only.
              </span>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => loadPlaylists(true)}
                  disabled={checkingRemote || syncing}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-mono font-medium rounded-lg bg-zinc-900 hover:bg-zinc-800 text-zinc-200 border border-zinc-800 hover:border-zinc-700 transition disabled:opacity-50 shadow-sm"
                >
                  <RefreshCw className={`w-3.5 h-3.5 text-zinc-400 ${checkingRemote ? 'animate-spin' : ''}`} />
                  {checkingRemote ? 'CHECKING...' : 'CHECK FOR NEW'}
                </button>
                <button
                  onClick={() => handleSync()}
                  disabled={syncing || checkingRemote}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-mono font-medium rounded-lg bg-red-600 hover:bg-red-500 text-white font-bold transition disabled:opacity-50 shadow-[0_0_15px_rgba(255,0,51,0.25)] hover:shadow-[0_0_20px_rgba(255,0,51,0.4)]"
                >
                  <Zap className={`w-3.5 h-3.5 ${syncing ? 'animate-spin' : ''}`} />
                  {syncing ? 'SYNCING...' : batchSize === 'all' ? 'SYNC ALL' : `SYNC NEXT ${batchSize}`}
                </button>
              </div>

              {/* Batch Size Selector */}
              <div className="flex items-center justify-between text-xs bg-zinc-900/80 px-2.5 py-1.5 rounded-lg border border-zinc-800 font-mono">
                <span className="text-zinc-400 text-[10px] tracking-wider uppercase">BATCH LIMIT:</span>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value === 'all' ? 'all' : Number(e.target.value))}
                  disabled={syncing}
                  className="bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 text-xs text-white focus:outline-none focus:border-red-500 font-mono"
                >
                  <option value={10}>10 videos</option>
                  <option value={28}>28 videos (Safe)</option>
                  <option value={50}>50 videos</option>
                  <option value="all">All unindexed</option>
                </select>
              </div>

              {/* Rate Limit Alert Banner in Sidebar */}
              {rateLimitInfo && (
                <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-500/50 text-amber-200 shadow-[0_0_20px_rgba(245,158,11,0.2)] space-y-2.5 font-mono">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <div className="p-1 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/40 shrink-0">
                        <AlertTriangle className="w-3.5 h-3.5" />
                      </div>
                      <div>
                        <h4 className="text-xs font-bold text-amber-300 tracking-tight flex items-center gap-1.5 font-tech">
                          [HTTP 429: RATE LIMITED]
                        </h4>
                        <p className="text-[10px] text-amber-400/80 font-sans">Sync paused to protect IP address</p>
                      </div>
                    </div>
                    <button
                      onClick={handleDismissRateLimit}
                      className="text-amber-400/70 hover:text-amber-200 p-1 rounded hover:bg-amber-900/50 transition shrink-0"
                      title="Dismiss"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="p-2.5 rounded-lg bg-black border border-amber-900/60 text-[10px] text-amber-200/90 leading-relaxed font-mono break-words">
                    {rateLimitInfo.message}
                    {rateLimitInfo.playlist && (
                      <div className="mt-1 text-[10px] text-red-400 font-mono">
                        PLAYLIST: {rateLimitInfo.playlist}
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-amber-300/80 pt-1 border-t border-amber-900/50">
                    <span className="flex items-center gap-1 text-[9px] text-zinc-400">
                      <Clock className="w-3 h-3 text-amber-400 shrink-0" />
                      {rateLimitInfo.timestamp ? `@ ${rateLimitInfo.timestamp}` : 'Cooldown active'}
                    </span>
                    <button
                      onClick={() => {
                        handleDismissRateLimit();
                        handleSync(rateLimitInfo.playlistId ? [rateLimitInfo.playlistId] : undefined);
                      }}
                      disabled={syncing}
                      className="px-2.5 py-1 rounded text-[10px] font-mono font-semibold bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/40 transition disabled:opacity-50"
                    >
                      RETRY SYNC
                    </button>
                  </div>
                </div>
              )}

              {/* Sync Progress Indicator */}
              {syncProgress && (
                <div className={`p-3 rounded-lg border font-mono ${
                  syncProgress.status === 'rate_limited'
                    ? 'bg-amber-950/40 border-amber-500/60 shadow-[0_0_15px_rgba(245,158,11,0.2)]'
                    : 'bg-zinc-900 border-zinc-800'
                }`}>
                  <div className="flex justify-between text-[11px] mb-1.5">
                    <span className={`flex items-center gap-1 font-semibold ${
                      syncProgress.status === 'rate_limited' ? 'text-amber-400' : 'text-zinc-200'
                    }`}>
                      <Radio className="w-3 h-3 text-red-500 animate-pulse" />
                      {syncProgress.status === 'rate_limited' ? 'RATE LIMIT PAUSED' : 'SYNC IN PROGRESS'}
                    </span>
                    <span className="text-zinc-300 font-bold">
                      {syncProgress.current} / {syncProgress.total}
                    </span>
                  </div>
                  <div className="w-full bg-black rounded-full h-1.5 overflow-hidden border border-zinc-800">
                    <div
                      className={`h-full transition-all duration-300 ${
                        syncProgress.status === 'rate_limited'
                          ? 'bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]'
                          : 'bg-gradient-to-r from-red-600 to-red-500 shadow-[0_0_8px_rgba(255,0,51,0.6)]'
                      }`}
                      style={{
                        width: `${syncProgress.total ? (syncProgress.current / syncProgress.total) * 100 : 5}%`,
                      }}
                    />
                  </div>
                  <p className="text-[10px] mt-2 truncate text-zinc-400">
                    &gt; {syncProgress.message}
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Playlists / Tracked List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          <div className="flex items-center justify-between px-1 mb-1">
            <h2 className="text-[10px] font-mono uppercase tracking-widest text-zinc-400 font-semibold flex items-center gap-1.5">
              <Database className="w-3 h-3 text-red-500" />
              Tracked Playlists ({playlists.length})
            </h2>
            <span className="text-[9px] font-mono text-emerald-400 font-semibold">ACTIVE</span>
          </div>

          {playlists.map((pl) => (
            <div
              key={pl.id}
              className={`p-3 rounded-xl bg-zinc-950 border transition space-y-2 ${
                rateLimitInfo?.playlistId === pl.id
                  ? 'border-amber-500/50 bg-amber-950/20'
                  : 'border-zinc-900 hover:border-zinc-800 hover:bg-zinc-900/40'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-xs font-semibold text-zinc-200 leading-snug line-clamp-1">{pl.name}</h3>
                {pl.url && (
                  <a
                    href={pl.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-zinc-500 hover:text-red-400 transition shrink-0"
                    title="View on YouTube"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>

              {/* Status Badges */}
              <div className="flex flex-wrap items-center gap-1.5 font-mono">
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-zinc-900 text-zinc-300 border border-zinc-800">
                  <CheckCircle2 className="w-2.5 h-2.5 text-emerald-400" />
                  {pl.indexed_count} INDEXED
                </span>
                {pl.new_count !== undefined && pl.new_count > 0 && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-red-950/60 text-red-300 border border-red-500/30 animate-pulse">
                    +{pl.new_count} NEW
                  </span>
                )}
                {rateLimitInfo?.playlistId === pl.id && (
                  <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-semibold bg-amber-950 text-amber-400 border border-amber-600">
                    THROTTLED
                  </span>
                )}
              </div>

              {/* Single Playlist Sync */}
              {!appConfig.read_only_mode && (
                <div className="pt-0.5">
                  <button
                    onClick={() => handleSync([pl.id])}
                    disabled={syncing}
                    className={`w-full py-1 text-[10px] font-mono rounded border transition disabled:opacity-50 flex items-center justify-center gap-1 ${
                      rateLimitInfo?.playlistId === pl.id
                        ? 'text-amber-300 hover:text-amber-100 bg-amber-950/40 hover:bg-amber-900/50 border-amber-800/60'
                        : 'text-zinc-400 hover:text-white hover:bg-zinc-900 border-zinc-800/80 hover:border-zinc-700'
                    }`}
                  >
                    <span>[Sync Playlist]</span>
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </aside>

      {/* MAIN PANEL: Workspace */}
      <main className="flex-1 flex flex-col h-full bg-black yt-grid overflow-hidden relative">
        {/* Rate Limit Alert Banner in Main Workspace */}
        {rateLimitInfo && (
          <div className="bg-amber-950/90 border-b border-amber-500/60 px-4 py-2.5 flex items-center justify-between gap-3 text-xs text-amber-200 shadow-[0_0_20px_rgba(245,158,11,0.25)] relative z-20 font-mono">
            <div className="flex items-center gap-2 overflow-hidden">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 animate-pulse" />
              <span className="font-bold text-amber-300 shrink-0 font-tech">[RATE_LIMIT_ACTIVE]</span>
              <span className="text-amber-200/90 truncate text-[11px]">
                YouTube rate limit reached (HTTP 429). Existing indexed video chunks remain fully searchable.
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={() => {
                  handleDismissRateLimit();
                  handleSync(rateLimitInfo.playlistId ? [rateLimitInfo.playlistId] : undefined);
                }}
                disabled={syncing}
                className="px-2.5 py-1 text-[10px] font-mono font-bold rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/50 transition disabled:opacity-50"
              >
                RETRY
              </button>
              <button
                onClick={handleDismissRateLimit}
                className="text-amber-400/80 hover:text-amber-200 p-1 rounded hover:bg-amber-900/40"
                title="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}

        {/* Top Control Bar */}
        <header className="p-3.5 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-xl flex flex-wrap items-center justify-between gap-3 relative z-10">
          {/* Target Scope Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono tracking-wider uppercase text-zinc-400 flex items-center gap-1 font-semibold">
              <Terminal className="w-3 h-3 text-red-500" />
              Scope:
            </span>
            <select
              value={selectedPlaylist}
              onChange={(e) => setSelectedPlaylist(e.target.value)}
              disabled={mode === 'freeform'}
              className={`bg-zinc-900 border border-zinc-800 rounded-lg px-2.5 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-red-500 font-mono transition ${
                mode === 'freeform' ? 'opacity-40 cursor-not-allowed' : ''
              }`}
              title={mode === 'freeform' ? 'Scope not applicable in Freeform mode' : undefined}
            >
              <option value="all">All Playlists ({totalIndexedVideos} videos indexed)</option>
              {playlists.map((pl) => (
                <option key={pl.id} value={pl.id}>
                  {pl.name} ({pl.indexed_count} indexed)
                </option>
              ))}
            </select>
          </div>

          {/* Mode Selector */}
          <div className="flex items-center gap-1 p-1 bg-zinc-900/90 rounded-xl border border-zinc-800 shadow-inner">
            {(['abhijit_ai', 'lookup', 'opinion', 'evolution', 'freeform'] as const).map((m) => {
              const item = modeLabels[m];
              const Icon = item.icon;
              const active = mode === m;
              return (
                <div key={m} className="relative group">
                  <button
                    type="button"
                    onClick={() => setMode(m)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-mono font-medium transition duration-200 ${
                      active
                        ? `${item.color} shadow-sm border font-semibold`
                        : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60 border border-transparent'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    <span>{item.label}</span>
                  </button>

                  {/* Hover Tooltip */}
                  <div
                    className={`absolute ${
                      m === 'abhijit_ai'
                        ? 'left-0'
                        : m === 'freeform'
                        ? 'right-0'
                        : 'left-1/2 -translate-x-1/2'
                    } top-full mt-2 hidden group-hover:flex flex-col items-center z-50 pointer-events-none`}
                  >
                    <div className="bg-zinc-900 text-zinc-200 border border-zinc-750 text-[11px] rounded-lg px-3 py-2 shadow-xl whitespace-nowrap font-sans">
                      <div className="font-tech text-white font-bold flex items-center gap-1.5">
                        <Icon className="w-3.5 h-3.5 text-red-500" />
                        {item.label}
                      </div>
                      <div className="text-zinc-400 text-[10px] mt-0.5 font-mono">{item.desc}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </header>

        {/* Content & Answers Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Default Empty State */}
          {!chatResponse && !loadingChat && !chatError && (
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto text-center py-8">
              {/* Central YouTube Red Glyph */}
              <div className="relative flex items-center justify-center w-20 h-20 mb-6">
                <div className="absolute inset-0 rounded-2xl border border-red-500/20 animate-ping opacity-30" />
                <div className="w-14 h-14 rounded-2xl bg-red-600/10 border border-red-500/30 text-red-500 flex items-center justify-center shadow-[0_0_25px_rgba(255,0,51,0.2)]">
                  <Sparkles className="w-7 h-7 text-red-500" />
                </div>
              </div>

              <h2 className="text-xl font-tech tracking-wider text-white font-bold mb-2">
                YouTube Transcript Search & Opinion Synthesis
              </h2>
              <p className="text-xs font-mono text-zinc-400 max-w-lg mb-8 leading-relaxed">
                Search timestamped transcripts across Abhijit Chavda's video archive. Synthesize opinions across years, locate exact quotes, or converse with Abhijit AI.
              </p>

              {/* Sample Queries Cards */}
              <div className="grid grid-cols-1 gap-2.5 w-full text-left font-mono">
                {[
                  {
                    q: 'What is your take on the multipolar world order and space exploration?',
                    tag: 'Abhijit AI',
                    mode: 'abhijit_ai' as const,
                  },
                  {
                    q: 'What does he say about String Theory and Quantum Mechanics?',
                    tag: 'Find Timestamp',
                    mode: 'lookup' as const,
                  },
                  {
                    q: 'How has his view on geopolitics and history changed over time?',
                    tag: 'Opinion Over Time',
                    mode: 'evolution' as const,
                  },
                  {
                    q: 'What is his overall philosophy on Indian education and history?',
                    tag: 'Overall Opinion',
                    mode: 'opinion' as const,
                  },
                  {
                    q: 'Explain quantum computing in simple terms',
                    tag: 'Freeform',
                    mode: 'freeform' as const,
                  },
                ].map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuestion(sample.q);
                      setMode(sample.mode);
                    }}
                    className="p-3 text-xs rounded-xl bg-zinc-950 border border-zinc-850 hover:border-zinc-700 hover:bg-zinc-900 text-zinc-300 hover:text-white transition duration-200 flex items-center justify-between group shadow-sm"
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span className="text-red-500 font-bold shrink-0">&gt;</span>
                      <span className="truncate">{sample.q}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-zinc-900 text-zinc-400 group-hover:text-white group-hover:border-zinc-700 border border-zinc-800 transition">
                        {sample.tag}
                      </span>
                      <CornerDownLeft className="w-3.5 h-3.5 text-zinc-600 group-hover:text-red-500 transition" />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Loading Indicator */}
          {loadingChat && !chatResponse && (
            <div className="p-8 max-w-3xl mx-auto rounded-2xl bg-zinc-950 border border-zinc-800 shadow-xl space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-3 h-3 rounded-full bg-red-500 animate-ping shrink-0" />
                <span className="text-xs font-mono tracking-wider text-red-400 font-semibold uppercase">
                  {mode === 'freeform'
                    ? 'Generating freeform response...'
                    : mode === 'abhijit_ai'
                    ? 'Synthesizing #AskAbhijit persona response...'
                    : 'Searching transcripts & synthesizing response...'}
                </span>
              </div>
              <div className="space-y-2">
                <div className="h-3 bg-zinc-800/80 rounded w-3/4 animate-pulse" />
                <div className="h-3 bg-zinc-800/80 rounded w-5/6 animate-pulse" />
                <div className="h-3 bg-zinc-800/80 rounded w-2/3 animate-pulse" />
              </div>
            </div>
          )}

          {/* Error Banner */}
          {chatError && (
            <div className="p-4 max-w-3xl mx-auto rounded-xl bg-rose-950/40 border border-rose-500/50 text-rose-200 flex items-start gap-3 text-sm font-mono shadow-lg">
              <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-bold text-rose-300">Synthesis Error</p>
                <p className="text-xs text-rose-300/90 mt-0.5">{chatError}</p>
              </div>
            </div>
          )}

          {/* Response Output Display */}
          {chatResponse && (
            <div className="max-w-3xl mx-auto space-y-6">
              {/* Synthesized Answer Card */}
              <div className="p-6 rounded-2xl bg-zinc-950 border border-zinc-800/90 hover:border-zinc-700 shadow-xl transition duration-300 space-y-4 relative yt-highlight">
                <div className="flex items-center justify-between pb-3 border-b border-zinc-800/80">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className={`px-2.5 py-0.5 rounded-md text-[10px] font-mono font-bold tracking-wider uppercase border ${
                      modeLabels[chatResponse.mode]?.badge || 'border-zinc-700 text-zinc-300 bg-zinc-900'
                    }`}>
                      {modeLabels[chatResponse.mode]?.label || chatResponse.mode}
                    </span>

                    {chatResponse.mode === 'freeform' ? (
                      <span className="text-[11px] font-mono text-zinc-400">
                        Freeform Synthesis
                      </span>
                    ) : chatResponse.mode === 'abhijit_ai' ? (
                      <span className="text-[11px] font-mono text-zinc-400">
                        #AskAbhijit POV • {chatResponse.sources.length} citation{chatResponse.sources.length !== 1 ? 's' : ''}
                      </span>
                    ) : (
                      <span className="text-[11px] font-mono text-zinc-400">
                        {chatResponse.sources.length} video citation{chatResponse.sources.length !== 1 ? 's' : ''} retrieved
                      </span>
                    )}

                    {loadingChat && (
                      <span className="inline-flex items-center gap-1.5 text-[10px] font-mono text-red-400 font-semibold animate-pulse ml-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-red-500 inline-block" />
                        Streaming response...
                      </span>
                    )}
                  </div>
                </div>

                {/* Markdown Renderer with Custom YouTube Timestamp Chips */}
                <div className="prose prose-invert prose-sm max-w-none text-zinc-200 leading-relaxed space-y-3 font-sans">
                  <ReactMarkdown
                    components={{
                      a: ({ node, href, children, ...props }) => {
                        const isYouTube = href?.includes('youtu');
                        return (
                          <a
                            href={href}
                            target="_blank"
                            rel="noreferrer"
                            className={`inline-flex items-center gap-1 font-mono font-medium transition ${
                              isYouTube
                                ? 'px-2 py-0.5 mx-0.5 rounded bg-red-950/60 hover:bg-red-900/80 text-red-300 hover:text-white border border-red-500/40 hover:border-red-400 shadow-[0_0_10px_rgba(255,0,51,0.2)] text-xs no-underline'
                                : 'text-zinc-300 hover:text-white hover:underline'
                            }`}
                            {...props}
                          >
                            {isYouTube && <Play className="w-2.5 h-2.5 fill-current text-red-500" />}
                            {children}
                          </a>
                        );
                      },
                      strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
                      code: ({ children }) => (
                        <code className="px-1.5 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-200 font-mono text-[11px]">
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {chatResponse.answer}
                  </ReactMarkdown>
                  {loadingChat && (
                    <span className="inline-block w-2 h-4 bg-red-500 animate-pulse ml-0.5 align-middle shadow-[0_0_8px_rgba(255,0,51,0.8)]" />
                  )}
                </div>
              </div>

              {/* Collapsible Evidence & Sources Drawer */}
              {chatResponse.sources.length > 0 && (
                <div className="rounded-2xl border border-zinc-800/80 bg-zinc-950/80 overflow-hidden shadow-lg">
                  <button
                    onClick={() => setShowSources(!showSources)}
                    className="w-full px-5 py-3 flex items-center justify-between text-xs font-mono tracking-wider uppercase text-zinc-400 hover:text-white hover:bg-zinc-900/50 transition"
                  >
                    <span className="flex items-center gap-2">
                      <Database className="w-3.5 h-3.5 text-red-500" />
                      Transcript Citations & Sources ({chatResponse.sources.length})
                    </span>
                    {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>

                  {showSources && (
                    <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3 border-t border-zinc-800/80 bg-black/50">
                      {chatResponse.sources.map((src, i) => (
                        <div
                          key={i}
                          className="p-3.5 rounded-xl bg-zinc-900/80 border border-zinc-800/90 hover:border-zinc-700 transition space-y-2 flex flex-col justify-between group"
                        >
                          <div>
                            <div className="flex items-start justify-between gap-2">
                              <h4 className="text-xs font-medium text-zinc-200 line-clamp-1 group-hover:text-white transition">
                                {src.title}
                              </h4>
                              <a
                                href={src.url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono bg-red-950/80 text-red-300 border border-red-500/40 hover:bg-red-900/60 hover:text-white transition shrink-0 shadow-sm"
                              >
                                <Play className="w-2.5 h-2.5 fill-current text-red-500" />
                                {src.timestamp_str}
                              </a>
                            </div>
                            <p className="text-[11px] text-zinc-400 line-clamp-3 mt-1.5 leading-relaxed italic border-l-2 border-red-500/40 pl-2">
                              "{src.snippet}"
                            </p>
                          </div>

                          <div className="flex items-center justify-between text-[10px] font-mono text-zinc-400 pt-2 border-t border-zinc-800/60">
                            <span>Date: {formatUploadDate(src.upload_date)}</span>
                            <span className="text-zinc-500">ID: {src.video_id}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Input Bar */}
        <footer className="p-4 border-t border-zinc-800/80 bg-zinc-950/90 backdrop-blur-xl relative z-10">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex items-center gap-2">
            <div className="relative flex-1">
              <div className="absolute left-3.5 top-1/2 -translate-y-1/2 text-red-500 font-mono font-bold text-xs pointer-events-none select-none">
                //&gt;
              </div>
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder={`Ask a question across indexed episodes (${modeLabels[mode].label})...`}
                disabled={loadingChat}
                className="w-full bg-zinc-900 border border-zinc-800 rounded-xl pl-9 pr-4 py-3 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-red-500 focus:ring-1 focus:ring-red-500 font-sans transition disabled:opacity-50 shadow-inner"
              />
            </div>
            <button
              type="submit"
              disabled={loadingChat || !question.trim()}
              className="px-5 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-mono font-bold text-xs tracking-wider transition disabled:opacity-30 shadow-[0_0_15px_rgba(255,0,51,0.25)] hover:shadow-[0_0_20px_rgba(255,0,51,0.45)] shrink-0 flex items-center gap-1.5"
            >
              <span>Ask</span>
              <CornerDownLeft className="w-3.5 h-3.5" />
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}
