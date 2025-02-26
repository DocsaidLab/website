import { useState } from 'react';
import { validateFileType } from '../utils/fileUtils';

/**
 * 自訂 Hook: 用於上傳 / 載入圖片並繪製到 originalCanvasRef，
 * 並自動處理縮放、警告、檔案存取等狀態。
 *
 * @param {object} params
 *   @param {React.RefObject<HTMLCanvasElement>} params.originalCanvasRef - 用來顯示圖片的 Canvas ref
 *   @param {object} params.warningMessage - 用來顯示各種警告字串的物件
 *       e.g. { imageTooLarge: '...', chooseFile: '...', invalidFileType: '...' }
 * @returns {object} - 內含 selectedFile, scale, imageInfo, warning, error, 以及 handleFileChange / handleExternalImageChange 等函式
 */
export function useImageUploader({ originalCanvasRef, warningMessage }) {
  // 狀態
  const [selectedFile, setSelectedFile] = useState(null);
  const [scale, setScale] = useState(1);
  const [imageInfo, setImageInfo] = useState(null);             // 畫布縮放後的寬高
  const [originalImageInfo, setOriginalImageInfo] = useState(null); // 圖片原始寬高
  const [warning, setWarning] = useState(null);
  const [error, setError] = useState(null);

  /**
   * 清除所有與圖片相關的狀態，並清除 originalCanvas 上的內容
   */
  const clearAll = () => {
    setSelectedFile(null);
    setScale(1);
    setImageInfo(null);
    setOriginalImageInfo(null);
    setWarning(null);
    setError(null);

    if (originalCanvasRef.current) {
      const ctx = originalCanvasRef.current.getContext('2d');
      ctx.clearRect(
        0,
        0,
        originalCanvasRef.current.width,
        originalCanvasRef.current.height
      );
    }
  };

  /**
   * 用於處理「本地檔案上傳」的函式
   * @param {File} file - 由 antd 或 <input type="file"> 選取的檔案
   * @returns {boolean} false: 告訴 antd 的 Upload 不要自行上傳
   */
  const handleFileChange = (file) => {
    if (!file) {
      setError(warningMessage?.chooseFile || 'Please choose a file first.');
      clearAll();
      return false; // antd 規定: 要阻止它自動上傳則 return false
    }

    if (!validateFileType(file)) {
      setError(warningMessage?.invalidFileType || 'Invalid file type.');
      clearAll();
      return false;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
      const img = new Image();
      img.onload = function () {
        drawImageOnCanvas(img, file.name, file.type);
      };
      img.src = event.target.result;
    };

    reader.readAsDataURL(file);
    return false;
  };

  /**
   * 用於處理「外部圖片 URL」的函式
   * @param {string} imageUrl - e.g. 'https://example.com/test.jpg'
   */
  const handleExternalImageChange = (imageUrl) => {
    // 先清除畫面
    clearAll();
    setError(null);
    setWarning(null);

    const canvas = originalCanvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'Anonymous'; // 若需要跨域讀取
    img.onload = function () {
      drawImageOnCanvas(img, 'external.jpg', 'image/jpeg');
    };
    img.src = imageUrl;
  };

  /**
   * 內部共用函式：將載入的 Image 繪製到 originalCanvas
   * 並根據大小決定是否縮放、是否顯示警告
   */
  const drawImageOnCanvas = (img, fileName, fileType) => {
    if (!originalCanvasRef.current) return;
    const canvas = originalCanvasRef.current;

    const originalWidth = img.width;
    const originalHeight = img.height;
    let scaleFactor = 1;
    if (img.width > 2000 || img.height > 2000) {
      scaleFactor = 2000 / Math.max(img.width, img.height);
    }
    setScale(scaleFactor);

    const scaledWidth = img.width * scaleFactor;
    const scaledHeight = img.height * scaleFactor;

    canvas.width = scaledWidth;
    canvas.height = scaledHeight;

    // 如果檔案超過 5000 px，顯示警告
    if (img.width > 5000 || img.height > 5000) {
      setWarning(warningMessage?.imageTooLarge || 'Image too large.');
    }

    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

    // 更新狀態
    setImageInfo({ width: scaledWidth, height: scaledHeight });
    setOriginalImageInfo({ width: originalWidth, height: originalHeight });

    // 產生 File 供後續上傳
    canvas.toBlob(
      function (blob) {
        if (!blob) return;
        const newFile = new File([blob], fileName, { type: fileType });
        setSelectedFile(newFile);
      },
      fileType,
      1 // 品質
    );
  };

  return {
    // 狀態
    selectedFile,
    scale,
    imageInfo,
    originalImageInfo,
    warning,
    error,
    // 狀態 setter (若你要在外部覆蓋它們)
    setWarning,
    setError,

    // 方法
    handleFileChange,
    handleExternalImageChange,
    clearAll,
  };
}
