import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useEffect, useState } from 'react';


export function useOpenCV() {
  const [openCvLoaded, setOpenCvLoaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadError, setDownloadError] = useState(null);

  useEffect(() => {
    if (!openCvLoaded) {
      downloadOpenCv();
    }
  }, []);

  const { siteConfig, i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  let opencvJsUrl;
  if (currentLocale === 'zh-hant') {
    opencvJsUrl = '/opencv.js';
  } else if (currentLocale === 'en') {
    opencvJsUrl = '/en/opencv.js';
  } else if (currentLocale === 'ja') {
    opencvJsUrl = '/ja/opencv.js';
  }

  const downloadOpenCv = () => {
    if (openCvLoaded || isDownloading) return;

    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadError(null);

    const xhr = new XMLHttpRequest();
    xhr.open('GET', opencvJsUrl, true);
    xhr.responseType = 'blob';

    xhr.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.floor((e.loaded / e.total) * 100);
        setDownloadProgress(percent);
      } else {
        setDownloadProgress(50);
      }
    };

    xhr.onload = () => {
      setIsDownloading(false);
      if (xhr.status === 200) {
        const blob = new Blob([xhr.response], { type: 'text/javascript' });
        const scriptUrl = URL.createObjectURL(blob);

        window.Module = {
          onRuntimeInitialized() {
            setOpenCvLoaded(true);
            console.log('OpenCV.js is ready from local /opencv.js');
          }
        };

        const script = document.createElement('script');
        script.src = scriptUrl;
        script.onload = () => {
          console.log('OpenCV.js script loaded (local).');
          if (window.cv && typeof window.cv.then === 'function') {
            window.cv.then((resolvedModule) => {
              window.cv = resolvedModule;
              setOpenCvLoaded(true);
            });
          } else {
            setOpenCvLoaded(true);
          }
        };
        script.onerror = (err) => {
          console.error('Failed to load OpenCV.js from object URL', err);
          setDownloadError('Failed to load OpenCV.js after fetching blob.');
        };
        document.body.appendChild(script);
      } else {
        setDownloadError(`Failed to download: status = ${xhr.status}`);
      }
    };

    xhr.onerror = () => {
      setIsDownloading(false);
      setDownloadError('Network error while downloading /opencv.js');
    };

    xhr.send();
  };

  return {
    openCvLoaded,
    isDownloading,
    downloadProgress,
    downloadError,
    downloadOpenCv
  };
}
