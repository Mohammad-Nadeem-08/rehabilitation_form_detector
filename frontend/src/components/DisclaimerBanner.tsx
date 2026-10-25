import React from 'react';
import { AlertCircle } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div style={{
      background: 'rgba(239, 68, 68, 0.08)',
      borderBottom: '1px solid rgba(239, 68, 68, 0.25)',
      padding: '8px 20px',
      fontSize: '0.8rem',
      color: '#FCA5A5',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      textAlign: 'center',
    }}>
      <AlertCircle size={15} color="#EF4444" style={{ flexShrink: 0 }} />
      <span>
        <strong>Educational Prototype Notice:</strong> Developed for 7th-semester deep learning evaluation. 
        Preliminary held-out test accuracy (82.5%) is research-only and <strong>not clinically validated</strong>. 
        Do not use for autonomous diagnostic or treatment decisions.
      </span>
    </div>
  );
};
