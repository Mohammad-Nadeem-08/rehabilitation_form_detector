import React, { useState, useEffect } from 'react';
import {
  History, Download, FileText, Search, Eye, Filter, RefreshCw
} from 'lucide-react';
import { fetchSessions, getReportDownloadUrl } from '../services/api';
import { SessionData } from '../types';
import { ReportModal } from '../components/ReportModal';

interface SessionHistoryPageProps {
  onSelectSession?: (session: SessionData) => void;
}

export const SessionHistoryPage: React.FC<SessionHistoryPageProps> = () => {
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [selectedSession, setSelectedSession] = useState<SessionData | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const list = await fetchSessions();
      setSessions(list);
    } catch (err) {
      console.error('Failed to load session history:', err);
    } finally {
      setLoading(false);
    }
  };

  const filteredSessions = sessions.filter((s) => {
    const term = searchTerm.toLowerCase();
    return (
      s.id.toLowerCase().includes(term) ||
      s.exercise_name.toLowerCase().includes(term) ||
      (s.notes && s.notes.toLowerCase().includes(term))
    );
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Info */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <span className="badge badge-teal">Persistence & Documentation</span>
              <span className="badge badge-green">SQLite Storage</span>
            </div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: '#F8FAFC' }}>
              Session History & Clinical PDF Reports
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', maxWidth: '780px' }}>
              Structured records of all recorded rehabilitation sessions, biomechanical angles, DTW deviations, and downloadable PDF reports.
            </p>
          </div>

          <button onClick={loadSessions} className="btn btn-secondary" style={{ padding: '8px 14px' }}>
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {/* Search Bar */}
      <div className="glass-panel" style={{ padding: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Search size={18} color="var(--text-secondary)" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by Session ID, exercise, or notes..."
            style={{
              width: '100%',
              background: 'transparent',
              border: 'none',
              outline: 'none',
              color: '#F8FAFC',
              fontSize: '0.9rem',
            }}
          />
        </div>
      </div>

      {/* Sessions Table */}
      <div className="glass-panel" style={{ padding: '24px' }}>
        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading recorded sessions...
          </div>
        ) : filteredSessions.length === 0 ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            No matching sessions found.
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', textAlign: 'left' }}>
                  <th style={{ padding: '10px 12px' }}>Session ID</th>
                  <th style={{ padding: '10px 12px' }}>Timestamp</th>
                  <th style={{ padding: '10px 12px' }}>Reference Standard</th>
                  <th style={{ padding: '10px 12px' }}>Reps</th>
                  <th style={{ padding: '10px 12px' }}>Duration</th>
                  <th style={{ padding: '10px 12px' }}>Peak ROM</th>
                  <th style={{ padding: '10px 12px' }}>Similarity Score</th>
                  <th style={{ padding: '10px 12px' }}>Model Status</th>
                  <th style={{ padding: '10px 12px', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredSessions.map((session) => (
                  <tr
                    key={session.id}
                    style={{ borderBottom: '1px solid rgba(51, 65, 85, 0.3)', transition: 'background 0.15s ease' }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(30, 41, 59, 0.4)')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <td style={{ padding: '12px', fontFamily: 'var(--font-mono)', color: '#00E5FF' }}>
                      {session.id}
                      {session.is_demo ? (
                        <span className="badge badge-amber" style={{ fontSize: '0.6rem', marginLeft: '6px' }}>DEMO</span>
                      ) : null}
                    </td>
                    <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                      {new Date(session.timestamp).toLocaleString([], {
                        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
                      })}
                    </td>
                    <td style={{ padding: '12px' }}>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#F8FAFC' }}>
                          {session.reference_id || 'ref-gold-standard-demo'}
                        </span>
                        <div>
                          {session.reference_type === 'patient_specific' ? (
                            <span className="badge" style={{ background: 'rgba(168, 85, 247, 0.2)', color: '#C084FC', border: '1px solid #A855F7', fontSize: '0.6rem' }}>
                              Patient-Specific
                            </span>
                          ) : (
                            <span className="badge badge-teal" style={{ fontSize: '0.6rem' }}>
                              Standard Baseline
                            </span>
                          )}
                          {session.patient_id && (
                            <span className="badge badge-blue" style={{ fontSize: '0.6rem', marginLeft: '4px' }}>
                              {session.patient_id}
                            </span>
                          )}
                        </div>
                      </div>
                    </td>
                    <td style={{ padding: '12px', fontWeight: 600 }}>{session.repetitions_completed}</td>
                    <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                      {session.duration_seconds.toFixed(1)}s
                    </td>
                    <td style={{ padding: '12px' }}>
                      {session.angle_stats?.peak_elevation?.toFixed(1) || 'N/A'}°
                    </td>
                    <td style={{ padding: '12px' }}>
                      <span className={`badge ${(session.comparison_score ?? 0) >= 80 ? 'badge-green' : 'badge-amber'}`}>
                        {session.comparison_score ? `${session.comparison_score.toFixed(1)}%` : 'N/A'}
                      </span>
                    </td>
                    <td style={{ padding: '12px' }}>
                      {session.lstm_prediction ? (
                        <span className="badge badge-teal">Class {session.lstm_prediction.predicted_class}</span>
                      ) : (
                        <span className="badge badge-amber" title="Live LSTM requires verified mocap adapter">Requires Validation</span>
                      )}
                    </td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>
                      <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                        <button
                          onClick={() => setSelectedSession(session)}
                          className="btn btn-secondary"
                          style={{ padding: '6px 10px', fontSize: '0.75rem' }}
                        >
                          <Eye size={12} /> View
                        </button>
                        <a
                          href={getReportDownloadUrl(session.id)}
                          target="_blank"
                          rel="noreferrer"
                          className="btn btn-primary"
                          style={{ padding: '6px 10px', fontSize: '0.75rem', textDecoration: 'none' }}
                        >
                          <Download size={12} /> PDF
                        </a>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedSession && (
        <ReportModal
          session={selectedSession}
          onClose={() => setSelectedSession(null)}
        />
      )}
    </div>
  );
};
