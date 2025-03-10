import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { useEffect, useState } from 'react';

export function useOpenCV() {
  const [openCvLoaded, setOpenCvLoaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadError, setDownloadError] = useState(null);

  const { i18n } = useDocusaurusContext();
  const currentLocale = i18n.currentLocale;

  let opencvJsUrl;
  if (currentLocale === 'zh-hant') {
    opencvJsUrl = '/opencv.js';
  } else if (currentLocale === 'en') {
    opencvJsUrl = '/en/opencv.js';
  } else if (currentLocale === 'ja') {
    opencvJsUrl = '/ja/opencv.js';
  } else {
    opencvJsUrl = '/opencv.js';
  }

  let opencvScriptAppended = false;

  useEffect(() => {
    if (openCvLoaded || window.cv?.Mat) {
      setOpenCvLoaded(true);
    } else {
      downloadOpenCv();
    }
  }, []);

  const downloadOpenCv = () => {
    if (openCvLoaded || isDownloading || window.cv?.Mat) return;
    if (opencvScriptAppended) return;

    opencvScriptAppended = true;

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

        if (!window.Module) { // 確保 Module 只初始化一次
          window.Module = {
            onRuntimeInitialized() {
              console.log('OpenCV.js is ready from local /opencv.js');
              setOpenCvLoaded(true);
            },
          };
        }

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
