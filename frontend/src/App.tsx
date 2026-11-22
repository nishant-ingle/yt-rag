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
  Key,
  Lock,
  Eye,
  EyeOff,
  MessageSquare,
  Bot,
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

export default function App() {
  const [appConfig, setAppConfig] = useState<AppConfig>({ read_only_mode: false, has_server_key: false });
  const [provider, setProvider] = useState<'openai' | 'gemini' | 'claude'>(() => {
    return (sessionStorage.getItem('byok_provider') as any) || 'openai';
  });
  const [apiKey, setApiKey] = useState<string>(() => {
    return sessionStorage.getItem('byok_key') || '';
  });
  const [showKey, setShowKey] = useState<boolean>(false);

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

  // Persist BYOK credentials in sessionStorage
  useEffect(() => {
    sessionStorage.setItem('byok_provider', provider);
  }, [provider]);

  useEffect(() => {
    if (apiKey) {
      sessionStorage.setItem('byok_key', apiKey);
    } else {
      sessionStorage.removeItem('byok_key');
    }
  }, [apiKey]);

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
      setSyncProgress({ current: 0, total: 1, message: 'Starting synchronization...', status: 'processing' });
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
            message: `⚠️ Rate limit exceeded: ${msg}`,
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
      // Refresh counts after sync
      await loadPlaylists(false);
      if (!encounteredRateLimit) {
        setRateLimitInfo(null);
      }
    } catch (err: any) {
      if (err.message?.includes('429') || err.message?.toLowerCase().includes('rate limit')) {
        setRateLimitInfo({
          message: `Sync not possible: ${err.message}`,
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
      setChatError(err.message || 'Error executing query');
    } finally {
      setLoadingChat(false);
    }
  }

  const modeLabels: Record<string, { label: string; desc: string; icon: any }> = {
    abhijit_ai: { label: 'Abhijit AI', desc: 'How Abhijit would hypothetically answer (#AskAbhijit)', icon: Bot },
    lookup: { label: 'Find Timestamp', desc: 'Direct quotes & exact moments', icon: Search },
    opinion: { label: 'Overall Opinion', desc: 'Synthesize consistent thesis', icon: Layers },
    evolution: { label: 'Opinion Over Time', desc: 'Chronological stance analysis', icon: History },
    freeform: { label: 'Freeform', desc: '#AskAbhijit Episode 9000', icon: MessageSquare },
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* LEFT SIDEBAR: Playlist Sync Manager */}
      <aside className="w-80 md:w-96 border-r border-slate-800 bg-slate-900/60 flex flex-col shrink-0">
        <div className="p-4 border-b border-slate-800">
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-red-600/20 text-red-500 flex items-center justify-center font-bold text-lg border border-red-500/30">
                ▶
              </div>
              <div>
                <h1 className="font-semibold text-base tracking-tight text-white">yt-rag engine</h1>
                <p className="text-xs text-slate-400">Abhijit Chavda Channel</p>
              </div>
            </div>
            {appConfig.read_only_mode && (
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-blue-950/80 text-blue-400 border border-blue-800/60 shrink-0">
                Read-Only
              </span>
            )}
          </div>
          {appConfig.read_only_mode && (
            <p className="text-[10px] text-blue-400/90 mt-1 font-medium">Hosted Snapshot (Read-Only)</p>
          )}
        </div>

        {/* BYOK Configuration Card */}
        <div className="p-3.5 border-b border-slate-800 bg-slate-900/40 space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
              <Key className="w-3.5 h-3.5 text-red-400" />
              LLM Provider
            </label>
            <span className="text-[10px] text-slate-500 font-mono">BYOK</span>
          </div>

          <select
            value={provider}
            onChange={(e) => setProvider(e.target.value as any)}
            className="w-full bg-slate-900 border border-slate-700/80 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-red-500 font-medium"
          >
            <option value="openai">OpenAI (gpt-4o-mini)</option>
            <option value="gemini" disabled className="text-slate-500">Gemini (Coming Soon)</option>
            <option value="claude" disabled className="text-slate-500">Claude (Coming Soon)</option>
          </select>

          <div className="relative">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              required
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={appConfig.has_server_key && !apiKey ? 'OpenAI Key (Server key active)' : 'OpenAI API Key (sk-...)'}
              className="w-full bg-slate-900 border border-slate-700/80 rounded-lg pl-2.5 pr-8 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-red-500 font-mono"
            />
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="absolute right-2 top-2 text-slate-400 hover:text-slate-200 transition"
              title={showKey ? 'Hide Key' : 'Show Key'}
            >
              {showKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            </button>
          </div>
          <p className="text-[10px] text-slate-500 leading-tight">
            Key kept in session memory; never saved to disk or DB.
          </p>
        </div>

        {/* Action Controls */}
        <div className="p-4 border-b border-slate-800 space-y-2.5">
          {appConfig.read_only_mode ? (
            <div className="p-2.5 rounded-lg bg-blue-950/40 border border-blue-800/50 text-blue-300 text-xs flex items-center gap-2">
              <Lock className="w-4 h-4 text-blue-400 shrink-0" />
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
                  className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${checkingRemote ? 'animate-spin' : ''}`} />
                  {checkingRemote ? 'Checking...' : 'Check for New'}
                </button>
                <button
                  onClick={() => handleSync()}
                  disabled={syncing || checkingRemote}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 text-xs font-medium rounded-lg bg-red-600 hover:bg-red-500 text-white transition disabled:opacity-50 shadow-sm shadow-red-900/50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${syncing ? 'animate-spin' : ''}`} />
                  {syncing ? 'Syncing...' : batchSize === 'all' ? 'Sync All New' : `Sync Next ${batchSize}`}
                </button>
              </div>

              {/* Batch Size Selector */}
              <div className="flex items-center justify-between text-xs bg-slate-950/60 px-2.5 py-1.5 rounded-lg border border-slate-800">
                <span className="text-slate-400 font-medium text-[11px]">Batch Limit:</span>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(e.target.value === 'all' ? 'all' : Number(e.target.value))}
                  disabled={syncing}
                  className="bg-slate-900 border border-slate-700 rounded px-2 py-0.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-red-500 font-medium"
                >
                  <option value={10}>10 videos</option>
                  <option value={28}>28 videos (Safe)</option>
                  <option value={50}>50 videos</option>
                  <option value="all">All unindexed</option>
                </select>
              </div>

              {/* Rate Limit Alert Banner in Sidebar */}
              {rateLimitInfo && (
                <div className="p-3.5 rounded-xl bg-amber-950/60 border border-amber-600/60 text-amber-200 shadow-lg space-y-2.5">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <div className="p-1.5 rounded-lg bg-amber-500/20 text-amber-400 border border-amber-500/30 shrink-0">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                      </div>
                      <div>
                        <h4 className="text-xs font-semibold text-amber-200 tracking-tight flex items-center gap-1.5">
                          Rate Limit Exceeded
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-amber-900/80 text-amber-300 border border-amber-700/60">
                            429
                          </span>
                        </h4>
                        <p className="text-[10px] text-amber-400/80">Sync paused to protect IP address</p>
                      </div>
                    </div>
                    <button
                      onClick={handleDismissRateLimit}
                      className="text-amber-400/70 hover:text-amber-200 p-1 rounded hover:bg-amber-900/50 transition shrink-0"
                      title="Dismiss warning"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>

                  <div className="p-2.5 rounded-lg bg-slate-950/70 border border-amber-900/60 text-xs text-amber-100/95 leading-relaxed font-mono text-[11px] break-words">
                    {rateLimitInfo.message}
                    {rateLimitInfo.playlist && (
                      <div className="mt-1 text-[10px] text-amber-400/90 font-sans">
                        Playlist: <span className="text-amber-200 font-medium">{rateLimitInfo.playlist}</span>
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-amber-300/80 pt-1 border-t border-amber-900/50">
                    <span className="flex items-center gap-1 text-[10px] text-slate-400">
                      <Clock className="w-3 h-3 text-amber-400 shrink-0" />
                      {rateLimitInfo.timestamp ? `Detected at ${rateLimitInfo.timestamp}` : 'Wait a while before retry'}
                    </span>
                    <button
                      onClick={() => {
                        handleDismissRateLimit();
                        handleSync(rateLimitInfo.playlistId ? [rateLimitInfo.playlistId] : undefined);
                      }}
                      disabled={syncing}
                      className="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/40 transition disabled:opacity-50"
                    >
                      Retry Sync
                    </button>
                  </div>
                </div>
              )}

              {/* Sync Progress Bar */}
              {syncProgress && (
                <div className={`mt-3 p-3 rounded-lg border ${
                  syncProgress.status === 'rate_limited'
                    ? 'bg-amber-950/50 border-amber-600/70'
                    : 'bg-slate-950/80 border-slate-800'
                }`}>
                  <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                    <span className={syncProgress.status === 'rate_limited' ? 'text-amber-400 font-medium flex items-center gap-1' : ''}>
                      {syncProgress.status === 'rate_limited' && <AlertTriangle className="w-3 h-3 text-amber-400" />}
                      {syncProgress.status === 'rate_limited' ? 'Rate Limit Paused' : 'Ingestion Progress'}
                    </span>
                    <span className="font-mono text-slate-300">
                      {syncProgress.current} / {syncProgress.total}
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-1.5 rounded-full transition-all duration-300 ${
                        syncProgress.status === 'rate_limited' ? 'bg-amber-500' : 'bg-red-500'
                      }`}
                      style={{
                        width: `${syncProgress.total ? (syncProgress.current / syncProgress.total) * 100 : 5}%`,
                      }}
                    />
                  </div>
                  <p className={`text-xs mt-2 truncate font-mono ${
                    syncProgress.status === 'rate_limited' ? 'text-amber-300' : 'text-slate-400'
                  }`}>
                    {syncProgress.message}
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Playlists List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 px-1">
            Tracked Playlists ({playlists.length})
          </h2>

          {playlists.map((pl) => (
            <div
              key={pl.id}
              className={`p-3.5 rounded-xl bg-slate-800/40 border transition space-y-2.5 ${
                rateLimitInfo?.playlistId === pl.id
                  ? 'border-amber-600/50 bg-amber-950/20'
                  : 'border-slate-800/80 hover:border-slate-700'
              }`}
            >
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-sm font-medium text-slate-200 leading-snug">{pl.name}</h3>
                {pl.url && (
                  <a
                    href={pl.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-slate-500 hover:text-slate-300 transition shrink-0"
                    title="View on YouTube"
                  >
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
              </div>

              {/* Badges */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
                  <CheckCircle2 className="w-3 h-3" />
                  Indexed: {pl.indexed_count}
                </span>
                {pl.new_count !== undefined && (
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium ${
                      pl.new_count > 0
                        ? 'bg-amber-950/60 text-amber-400 border border-amber-800/60'
                        : 'bg-slate-800/60 text-slate-400 border border-slate-700/60'
                    }`}
                  >
                    New: {pl.new_count}
                  </span>
                )}
                {rateLimitInfo?.playlistId === pl.id && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium bg-amber-950/80 text-amber-400 border border-amber-700/70">
                    <AlertTriangle className="w-3 h-3 text-amber-400" />
                    Rate Limited
                  </span>
                )}
              </div>

              {/* Single Playlist Sync */}
              {!appConfig.read_only_mode && (
                <div className="pt-1">
                  <button
                    onClick={() => handleSync([pl.id])}
                    disabled={syncing}
                    className={`w-full py-1 text-[11px] rounded border transition disabled:opacity-50 ${
                      rateLimitInfo?.playlistId === pl.id
                        ? 'text-amber-300 hover:text-amber-100 bg-amber-950/40 hover:bg-amber-900/50 border-amber-800/60'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/80 border-slate-800'
                    }`}
                  >
                    {rateLimitInfo?.playlistId === pl.id
                      ? 'Retry Sync (Rate Limited)'
                      : batchSize === 'all'
                      ? 'Sync This Playlist'
                      : `Sync Next ${batchSize} Videos`}
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </aside>

      {/* MAIN PANEL: RAG Workspace */}
      <main className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden">
        {/* Rate Limit Notification Banner in Main Workspace */}
        {rateLimitInfo && (
          <div className="bg-amber-950/90 border-b border-amber-800/70 px-4 py-2.5 flex items-center justify-between gap-3 text-xs text-amber-200 shadow-md">
            <div className="flex items-center gap-2 overflow-hidden">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span className="font-semibold text-amber-300 shrink-0">Rate Limit Exceeded:</span>
              <span className="text-amber-200/90 truncate">
                Sync is paused because YouTube rate limits have been reached (HTTP 429). Existing indexed videos remain fully searchable.
              </span>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                onClick={() => {
                  handleDismissRateLimit();
                  handleSync(rateLimitInfo.playlistId ? [rateLimitInfo.playlistId] : undefined);
                }}
                disabled={syncing}
                className="px-2 py-1 text-[11px] font-medium rounded bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/40 transition disabled:opacity-50"
              >
                Retry Sync
              </button>
              <button
                onClick={handleDismissRateLimit}
                className="text-amber-400/80 hover:text-amber-200 p-0.5 rounded hover:bg-amber-900/40"
                title="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        )}
        {/* Controls Bar */}
        <header className="p-4 border-b border-slate-800 bg-slate-900/30 flex flex-wrap items-center justify-between gap-3 relative z-10">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-medium">Scope:</span>
            <select
              value={selectedPlaylist}
              onChange={(e) => setSelectedPlaylist(e.target.value)}
              disabled={mode === 'freeform'}
              className={`bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-red-500 ${
                mode === 'freeform' ? 'opacity-50 cursor-not-allowed' : ''
              }`}
              title={mode === 'freeform' ? 'Playlist scope is not applicable in Freeform mode' : undefined}
            >
              <option value="all">All Playlists ({playlists.reduce((acc, p) => acc + p.indexed_count, 0)} videos)</option>
              {playlists.map((pl) => (
                <option key={pl.id} value={pl.id}>
                  {pl.name} ({pl.indexed_count} videos)
                </option>
              ))}
            </select>
          </div>

          {/* Mode Selector Tabs */}
          <div className="flex items-center gap-1 p-1 bg-slate-900/80 rounded-xl border border-slate-800">
            {(['abhijit_ai', 'lookup', 'opinion', 'evolution', 'freeform'] as const).map((m) => {
              const item = modeLabels[m];
              const Icon = item.icon;
              const active = mode === m;
              return (
                <div key={m} className="relative group">
                  <button
                    type="button"
                    onClick={() => setMode(m)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                      active
                        ? 'bg-red-600 text-white shadow-sm shadow-red-900/50'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                    {item.label}
                  </button>
                  {/* Floating Mode Tooltip */}
                  <div
                    className={`absolute ${
                      m === 'abhijit_ai'
                        ? 'left-0'
                        : m === 'freeform'
                        ? 'right-0'
                        : 'left-1/2 -translate-x-1/2'
                    } top-full mt-2 hidden group-hover:flex flex-col items-center z-50 pointer-events-none`}
                  >
                    <div
                      className={`w-2 h-2 -mb-1 rotate-45 bg-slate-800 border-l border-t border-slate-700 ${
                        m === 'abhijit_ai'
                          ? 'self-start ml-6'
                          : m === 'freeform'
                          ? 'self-end mr-6'
                          : ''
                      }`}
                    />
                    <div className="bg-slate-800 text-slate-200 border border-slate-700 text-[11px] rounded-lg px-2.5 py-1.5 shadow-xl whitespace-nowrap font-normal">
                      <div className="font-semibold text-slate-100 flex items-center gap-1">
                        <Icon className="w-3 h-3 text-red-400" />
                        {item.label}
                      </div>
                      <div className="text-slate-400 text-[10px] mt-0.5">{item.desc}</div>
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
            <div className="h-full flex flex-col items-center justify-center max-w-xl mx-auto text-center py-12 text-slate-400">
              <div className="w-14 h-14 rounded-2xl bg-red-600/10 border border-red-500/20 text-red-400 flex items-center justify-center mb-4">
                <Sparkles className="w-7 h-7" />
              </div>
              <h2 className="text-lg font-semibold text-slate-200 mb-1">YouTube Synthesis & Timestamp Deep Search</h2>
              <p className="text-xs text-slate-400 mb-6">
                Ask specific questions to lookup timestamps, request an overall opinion synthesis, analyze opinion evolution over time, ask anything freely, or hear how Abhijit Chavda would hypothetically answer.
              </p>

              {/* Sample Queries */}
              <div className="grid grid-cols-1 gap-2 w-full text-left">
                {[
                  'What is your take on the multipolar world order and space exploration? (Abhijit AI)',
                  'What does he say about String Theory and Quantum Mechanics?',
                  'How has his view on geopolitics and history changed over time?',
                  'What is his overall philosophy on Indian education and history?',
                  'Explain quantum computing in simple terms (Freeform)',
                ].map((sample, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setQuestion(sample);
                    }}
                    className="p-3 text-xs rounded-xl bg-slate-900/70 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-white transition flex items-center justify-between group"
                  >
                    <span>{sample}</span>
                    <span className="text-slate-500 group-hover:text-red-400 transition">↵</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Loading State: show when loading and no streamed response yet */}
          {loadingChat && !chatResponse && (
            <div className="p-8 max-w-3xl mx-auto rounded-2xl bg-slate-900/40 border border-slate-800/80 animate-pulse space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-4 h-4 rounded-full bg-red-500 animate-ping" />
                <span className="text-sm text-slate-300 font-medium">
                  {mode === 'freeform'
                    ? 'Generating freeform response...'
                    : mode === 'abhijit_ai'
                    ? 'Synthesizing response in Abhijit Chavda persona (#AskAbhijit)...'
                    : 'Synthesizing transcripts with hybrid vector + FTS retrieval...'}
                </span>
              </div>
              <div className="h-4 bg-slate-800 rounded w-3/4" />
              <div className="h-4 bg-slate-800 rounded w-5/6" />
              <div className="h-4 bg-slate-800 rounded w-2/3" />
            </div>
          )}

          {/* Error State */}
          {chatError && (
            <div className="p-4 max-w-3xl mx-auto rounded-xl bg-red-950/40 border border-red-800/60 text-red-200 flex items-start gap-3 text-sm">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-red-300">Query Failed</p>
                <p className="text-xs text-red-400/90 mt-0.5">{chatError}</p>
              </div>
            </div>
          )}

          {/* Response Display */}
          {chatResponse && (
            <div className="max-w-3xl mx-auto space-y-6">
              {/* Answer Card */}
              <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800 shadow-xl space-y-4">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <span className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold tracking-wide uppercase bg-red-500/10 text-red-400 border border-red-500/20">
                      Mode: {modeLabels[chatResponse.mode]?.label || chatResponse.mode}
                    </span>
                    {chatResponse.mode === 'freeform' ? (
                      <span className="text-xs text-slate-400">
                        Open inquiry (no video constraints)
                      </span>
                    ) : chatResponse.mode === 'abhijit_ai' ? (
                      <span className="text-xs text-slate-400">
                        #AskAbhijit Persona {chatResponse.sources.length > 0 ? `(${chatResponse.sources.length} video citations)` : ''}
                      </span>
                    ) : (
                      <span className="text-xs text-slate-400">
                        {chatResponse.sources.length} citations retrieved
                      </span>
                    )}
                    {loadingChat && (
                      <span className="inline-flex items-center gap-1 text-[11px] text-amber-400 font-medium animate-pulse ml-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block" />
                        Generating...
                      </span>
                    )}
                  </div>
                </div>

                {/* Markdown Renderer with Custom Links */}
                <div className="prose prose-invert prose-sm max-w-none text-slate-200 leading-relaxed space-y-3">
                  <ReactMarkdown
                    components={{
                      a: ({ node, href, children, ...props }) => {
                        const isYouTube = href?.includes('youtu');
                        return (
                          <a
                            href={href}
                            target="_blank"
                            rel="noreferrer"
                            className={`inline-flex items-center gap-1 font-medium transition ${
                              isYouTube
                                ? 'px-2 py-0.5 mx-0.5 rounded-md bg-red-950/50 hover:bg-red-900/60 text-red-400 hover:text-red-300 border border-red-800/50 text-xs no-underline'
                                : 'text-blue-400 hover:underline'
                            }`}
                            {...props}
                          >
                            {isYouTube && <Play className="w-2.5 h-2.5 fill-current" />}
                            {children}
                          </a>
                        );
                      },
                    }}
                  >
                    {chatResponse.answer}
                  </ReactMarkdown>
                  {loadingChat && (
                    <span className="inline-block w-2 h-4 bg-red-500 animate-pulse ml-0.5 align-middle" />
                  )}
                </div>
              </div>

              {/* Collapsible Sources Drawer */}
              {chatResponse.sources.length > 0 && (
                <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 overflow-hidden">
                  <button
                    onClick={() => setShowSources(!showSources)}
                    className="w-full px-5 py-3.5 flex items-center justify-between text-xs font-semibold uppercase tracking-wider text-slate-400 hover:text-slate-200 hover:bg-slate-800/30 transition"
                  >
                    <span>Cited Transcript Sources ({chatResponse.sources.length})</span>
                    {showSources ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>

                  {showSources && (
                    <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-3 border-t border-slate-800/60">
                      {chatResponse.sources.map((src, i) => (
                        <div
                          key={i}
                          className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80 hover:border-slate-700 transition space-y-2 flex flex-col justify-between"
                        >
                          <div>
                            <div className="flex items-start justify-between gap-2">
                              <h4 className="text-xs font-medium text-slate-200 line-clamp-1">
                                {src.title}
                              </h4>
                              <a
                                href={src.url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono bg-red-950/80 text-red-400 border border-red-800/60 hover:bg-red-900/60 transition shrink-0"
                              >
                                <Play className="w-2.5 h-2.5 fill-current" />
                                {src.timestamp_str}
                              </a>
                            </div>
                            <p className="text-[11px] text-slate-400 line-clamp-3 mt-1.5 leading-relaxed italic">
                              "{src.snippet}"
                            </p>
                          </div>

                          <div className="flex items-center justify-between text-[10px] text-slate-500 pt-2 border-t border-slate-900">
                            <span>Date: {src.upload_date || 'N/A'}</span>
                            <span className="font-mono">ID: {src.video_id}</span>
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
        <footer className="p-4 border-t border-slate-800 bg-slate-900/60">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto flex items-center gap-2">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder={`Ask a question (Mode: ${modeLabels[mode].label})...`}
              disabled={loadingChat}
              className="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-red-500 focus:border-red-500 transition disabled:opacity-50"
            >
            </input>
            <button
              type="submit"
              disabled={loadingChat || !question.trim()}
              className="px-5 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-medium text-sm transition disabled:opacity-40 shadow-sm shadow-red-900/50 shrink-0"
            >
              Ask
            </button>
          </form>
        </footer>
      </main>
    </div>
  );
}
