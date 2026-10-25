import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { DashboardPage } from './pages/DashboardPage';
import { ExerciseReferencePage } from './pages/ExerciseReferencePage';
import { LiveAssessmentPage } from './pages/LiveAssessmentPage';
import { ModelLabPage } from './pages/ModelLabPage';
import { SessionHistoryPage } from './pages/SessionHistoryPage';
import { ReportModal } from './components/ReportModal';
import { checkHealth } from './services/api';
import { SessionData } from './types';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [backendHealthy, setBackendHealthy] = useState<boolean>(false);
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [activeReferenceId, setActiveReferenceId] = useState<string | null>('ref-gold-standard-demo');
  const [inspectSession, setInspectSession] = useState<SessionData | null>(null);

  useEffect(() => {
    const verifyBackend = async () => {
      try {
        const health = await checkHealth();
        setBackendHealthy(health.status === 'healthy');
        setModelLoaded(health.model_loaded);
      } catch (err) {
        setBackendHealthy(false);
        setModelLoaded(false);
      }
    };

    verifyBackend();
    const interval = setInterval(verifyBackend, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', backgroundColor: 'var(--bg-deep)' }}>
      {/* Educational Prototype Disclaimer Banner */}
      <DisclaimerBanner />

      {/* Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendHealthy={backendHealthy}
        modelLoaded={modelLoaded}
      />

      {/* Main Content View */}
      <main style={{ flex: 1, padding: '24px 20px', maxWidth: '1360px', width: '100%', margin: '0 auto' }}>
        {activeTab === 'dashboard' && (
          <DashboardPage
            setActiveTab={setActiveTab}
            onSelectSession={(s) => setInspectSession(s)}
          />
        )}

        {activeTab === 'reference' && (
          <ExerciseReferencePage
            activeReferenceId={activeReferenceId}
            setActiveReferenceId={setActiveReferenceId}
          />
        )}

        {activeTab === 'assessment' && (
          <LiveAssessmentPage
            activeReferenceId={activeReferenceId}
            setActiveTab={setActiveTab}
            onSessionSaved={(s) => setInspectSession(s)}
          />
        )}

        {activeTab === 'model-lab' && (
          <ModelLabPage />
        )}

        {activeTab === 'history' && (
          <SessionHistoryPage
            onSelectSession={(s) => setInspectSession(s)}
          />
        )}
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--border-subtle)',
        padding: '20px',
        textAlign: 'center',
        fontSize: '0.78rem',
        color: 'var(--text-muted)',
        background: 'rgba(10, 17, 40, 0.9)',
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <strong>RehabVision AI</strong> — 7th-Semester Deep Learning Project. Developed by Full-Stack & ML Engineering Team.
          </div>
          <div>
            PyTorch LSTM • MediaPipe Vision • Dynamic Time Warping • SQLite Persistence
          </div>
        </div>
      </footer>

      {/* Inspect Session Modal */}
      {inspectSession && (
        <ReportModal
          session={inspectSession}
          onClose={() => setInspectSession(null)}
        />
      )}
    </div>
  );
};

export default App;
