import { useEffect, useState } from 'react';

export function useApiStatus(apiUrl) {
  const [apiStatus, setApiStatus] = useState(null);

  useEffect(() => {
    checkApiStatus();
  }, [apiUrl]);

  const checkApiStatus = async () => {
    try {
      const response = await fetch(apiUrl);
      if (response.ok) {
        setApiStatus('online');
      } else {
        setApiStatus('offline');
      }
    } catch (err) {
      setApiStatus('offline');
      console.error('Error checking API status:', err);
    }
  };

  return { apiStatus };
}
