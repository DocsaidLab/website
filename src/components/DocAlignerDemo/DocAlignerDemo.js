import { ClearOutlined, DownloadOutlined, UploadOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  InputNumber,
  Progress,
  Row,
  Space,
  Spin,
  Typography,
  Upload
} from 'antd';
import React, { useEffect, useRef, useState } from 'react';
import styles from './DocAlignerDemo.module.css';

const { Title, Text } = Typography;

const DocAlignerDemo = ({
  chooseFileLabel,
  uploadButtonLabel,
  downloadButtonLabel,
  clearButtonLabel,
  processingMessage,
  errorMessage,
  warningMessage,
  imageInfoTitle,
  inferenceInfoTitle,
  polygonInfoTitle,
  inferenceTimeLabel,
  timestampLabel,
  fileNameLabel,
  fileSizeLabel,
  fileTypeLabel,
  imageSizeLabel,
  TransformedTitle,
  TransformedWidthLabel,
  TransformedHeightLabel,
  TransformedButtonLabel,
  externalImage,
}) => {
  const fileInputRef = useRef(null);
  const originalCanvasRef = useRef(null);
  const processedCanvasRef = useRef(null);
  const transformedCanvasRef = useRef(null);

  // ------ 狀態管理 ------
  const [predictionData, setPredictionData] = useState(null);
  const [predictionDataScale, setPredictionDataScale] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [imageInfo, setImageInfo] = useState(null);
  const [originalImageInfo, setOriginalImageInfo] = useState(null);
  const [scale, setScale] = useState(1);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [timestamp, setTimestamp] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);
  const [outputWidth, setOutputWidth] = useState(768);
  const [outputHeight, setOutputHeight] = useState(480);
  const [apiStatus, setApiStatus] = useState(null);

  // ------ OpenCV 下載與載入狀態 ------
  const [openCvLoaded, setOpenCvLoaded] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadError, setDownloadError] = useState(null);

  // 若外部傳入影像路徑，則自動載入
  useEffect(() => {
    if (externalImage) {
      handleExternalImageChange(externalImage);
    }
  }, [externalImage]);

  // 檢查後端 API 狀態
  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch('https://api.docsaid.org/docaligner-predict');
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

  // 頁面載入時，就自動嘗試下載 OpenCV
  useEffect(() => {
    if (!openCvLoaded) {
      downloadOpenCv();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 若成功取得 polygon 且已載入 OpenCV，執行攤平
  useEffect(() => {
    if (predictionData && openCvLoaded) {
      performPerspectiveTransform();
    }
  }, [predictionData, openCvLoaded, outputWidth, outputHeight]);

  // ------ 載入圖片 (external or local) ------
  const handleExternalImageChange = (imageSource) => {
    clearAll();
    setError(null);
    setWarning(null);
    setPredictionData(null);
    setPredictionDataScale(null);
    setImageInfo(null);

    const canvas = originalCanvasRef.current;
    if (!canvas) return;

    const img = new Image();
    img.crossOrigin = 'Anonymous';
    img.onload = function () {
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

      if (img.width > 5000 || img.height > 5000) {
        setWarning(warningMessage.imageTooLarge);
      }

      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

      setImageInfo({ width: scaledWidth, height: scaledHeight });
      setOriginalImageInfo({ width: originalWidth, height: originalHeight });

      canvas.toBlob(function (blob) {
        const file = new File([blob], 'example.jpg', { type: 'image/jpeg' });
        setSelectedFile(file);
      }, 'image/jpeg');
    };
    img.src = imageSource;
  };

  const validateFileType = (file) => {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    return validTypes.includes(file.type);
  };

  const handleFileChange = (file) => {
    if (!file) {
      setError(errorMessage.chooseFile);
      clearAll();
      return false;
    }
    if (!validateFileType(file)) {
      setError(errorMessage.invalidFileType);
      clearAll();
      return false;
    }

    const reader = new FileReader();
    reader.onload = function (event) {
      const img = new Image();
      img.onload = function () {
        const canvas = originalCanvasRef.current;
        if (!canvas) return;
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

        if (img.width > 5000 || img.height > 5000) {
          setWarning(warningMessage.imageTooLarge);
        }

        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

        setImageInfo({ width: scaledWidth, height: scaledHeight });
        setOriginalImageInfo({ width: originalWidth, height: originalHeight });

        canvas.toBlob(function (blob) {
          const scaledFile = new File([blob], file.name, { type: file.type });
          setSelectedFile(scaledFile);
        }, file.type);
      };
      img.src = event.target.result;
    };

    reader.readAsDataURL(file);
    return false;
  };

  // ------ 清除結果 ------
  const clearAll = () => {
    setPredictionData(null);
    setPredictionDataScale(null);
    setWarning(null);
    setImageInfo(null);
    setOriginalImageInfo(null);
    setSelectedFile(null);
    setInferenceTime(0);
    setScale(1);
    setOutputWidth(768);
    setOutputHeight(480);

    if (originalCanvasRef.current) {
      const originalCtx = originalCanvasRef.current.getContext('2d');
      originalCtx.clearRect(0, 0, originalCanvasRef.current.width, originalCanvasRef.current.height);
    }

    if (processedCanvasRef.current) {
      const processedCtx = processedCanvasRef.current.getContext('2d');
      processedCtx.clearRect(0, 0, processedCanvasRef.current.width, processedCanvasRef.current.height);
    }

    if (transformedCanvasRef.current) {
      const transformedCtx = transformedCanvasRef.current.getContext('2d');
      transformedCtx.clearRect(0, 0, transformedCanvasRef.current.width, transformedCanvasRef.current.height);
    }
  };

  // ------ 上傳影像到後端預測 ------
  const uploadImage = () => {
    const originalCanvas = originalCanvasRef.current;
    const processedCanvas = processedCanvasRef.current;

    if (!originalCanvas || !processedCanvas) return;
    let file = selectedFile;

    if (!file) {
      setError(errorMessage.chooseFile);
      clearAll();
      return;
    }

    if (!validateFileType(file)) {
      setError(errorMessage.invalidFileType);
      clearAll();
      return;
    }

    setIsLoading(true);
    setError(null);
    setWarning(null);
    setPredictionData(null);
    setPredictionDataScale(null);

    const formData = new FormData();
    formData.append('file', file);

    fetch('https://api.docsaid.org/docaligner-predict', {
      method: 'POST',
      body: formData
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(errorMessage.networkError);
        }
        return response.json();
      })
      .then((data) => {
        if (!data.polygon || data.polygon.length === 0) {
          setWarning(warningMessage.noPolygon);
          setIsLoading(false);
          return;
        }

        setInferenceTime(data.inference_time);
        setTimestamp(data.timestamp);
        setPredictionData(data);

        // 將伺服器回傳的 polygon，等比縮放回到在原始 canvas 的座標
        const adjustedPolygon = data.polygon.map((point) => [
          point[0] / scale,
          point[1] / scale,
        ]);
        setPredictionDataScale({ ...data, polygon: adjustedPolygon });

        const ctx = processedCanvas.getContext('2d');
        ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
        processedCanvas.width = originalCanvas.width;
        processedCanvas.height = originalCanvas.height;
        ctx.drawImage(originalCanvas, 0, 0);
        drawPolygon(ctx, data.polygon);
      })
      .catch((err) => {
        console.error('Error:', err);
        setError(`${errorMessage.uploadError}: ${err.message || err}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  // ------ 繪製多邊形與箭頭 ------
  const drawPolygon = (ctx, polygon) => {
    const colors = [
      [0, 255, 255],
      [0, 255, 0],
      [255, 0, 0],
      [255, 255, 0],
    ];
    const sortedPolygon = sortPolygonClockwise(polygon);
    const numPoints = sortedPolygon.length;

    sortedPolygon.forEach((p1, i) => {
      const p2 = sortedPolygon[(i + 1) % numPoints];
      const color = `rgb(${colors[i % colors.length].join(',')})`;
      const thickness = Math.max(ctx.canvas.width * 0.005, 1);

      // 畫點
      ctx.beginPath();
      ctx.arc(p1[0], p1[1], thickness * 2, 0, Math.PI * 2, false);
      ctx.fillStyle = color;
      ctx.fill();

      // 畫箭頭
      drawArrow(ctx, p1[0], p1[1], p2[0], p2[1], thickness, color);
    });
  };

  const sortPolygonClockwise = (polygon) => {
    const sortedByY = polygon.slice().sort((a, b) => a[1] - b[1]);
    const topPoints = sortedByY.slice(0, 2);
    const bottomPoints = sortedByY.slice(2);
    const [topLeft, topRight] = topPoints.sort((a, b) => a[0] - b[0]);
    const [bottomLeft, bottomRight] = bottomPoints.sort((a, b) => a[0] - b[0]);
    return [topLeft, topRight, bottomRight, bottomLeft];
  };

  const drawArrow = (ctx, fromX, fromY, toX, toY, thickness, color) => {
    const headlen = thickness * 5;
    const angle = Math.atan2(toY - fromY, toX - fromX);
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(
      toX - headlen * Math.cos(angle - Math.PI / 6),
      toY - headlen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      toX - headlen * Math.cos(angle + Math.PI / 6),
      toY - headlen * Math.sin(angle + Math.PI / 6)
    );
    ctx.lineTo(toX, toY);
    ctx.fillStyle = color;
    ctx.fill();
  };

  // ------ 下載或導出結果 ------
  const downloadJSON = () => {
    if (!predictionData) return;
    const dataStr =
      'data:text/json;charset=utf-8,' +
      encodeURIComponent(JSON.stringify(predictionData));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute('href', dataStr);
    downloadAnchorNode.setAttribute('download', 'prediction.json');
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const downloadTransformedImage = () => {
    const transformedCanvas = transformedCanvasRef.current;
    if (!transformedCanvas) return;

    const dataURL = transformedCanvas.toDataURL('image/jpeg');
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'transformed_image.jpg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // ------ 下載本地端 /opencv.js (顯示進度) ------
  const downloadOpenCv = () => {
    if (openCvLoaded || isDownloading) return; // 已載入或正在下載中，不要重複執行

    setIsDownloading(true);
    setDownloadProgress(0);
    setDownloadError(null);

    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/opencv.js', true);
    xhr.responseType = 'blob'; // 以 Blob 形式接收

    // 進度事件
    xhr.onprogress = (e) => {
      if (e.lengthComputable) {
        const percent = Math.floor((e.loaded / e.total) * 100);
        setDownloadProgress(percent);
      } else {
        // 若無法計算長度，就顯示個大約值或保留 0
        setDownloadProgress(50);
      }
    };

    xhr.onload = () => {
      if (xhr.status === 200) {
        // 將 Blob 轉成 ObjectURL 再動態插 <script>
        const blob = new Blob([xhr.response], { type: 'text/javascript' });
        const scriptUrl = URL.createObjectURL(blob);

        // OpenCV 初始化 callback
        window.Module = {
          onRuntimeInitialized() {
            console.log('OpenCV.js is ready from local /opencv.js');
            setOpenCvLoaded(true);
          },
        };

        const script = document.createElement('script');
        script.src = scriptUrl;
        script.onload = () => {
          console.log('OpenCV.js script loaded (local).');
          // 若 window.cv 是 Promise，就等它 resolve
          if (window.cv && typeof window.cv.then === 'function') {
            window.cv.then((resolvedModule) => {
              console.log('Resolved the Promise-based cv module', resolvedModule);
              window.cv = resolvedModule;
              setOpenCvLoaded(true);
            });
          } else {
            // 不是 Promise 版 => 直接可用
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
      setIsDownloading(false);
    };

    xhr.onerror = () => {
      setDownloadError('Network error while downloading /opencv.js');
      setIsDownloading(false);
    };

    xhr.send();
  };

  // ------ 執行透視轉換 ------
  const performPerspectiveTransform = () => {
    if (!predictionData || !openCvLoaded || !window.cv || !window.cv.imread) {
      console.log('Cannot transform yet: missing data or OpenCV not loaded.');

      // detail info
      console.log('predictionData:', predictionData);
      console.log('openCvLoaded:', openCvLoaded);
      console.log('window.cv:', window.cv);
      console.log('window.cv.imread:', window.cv?.imread);

      return;
    }

    const originalCanvas = originalCanvasRef.current;
    const transformedCanvas = transformedCanvasRef.current;
    if (!originalCanvas || !transformedCanvas) {
      console.log('Canvas references not ready.');
      return;
    }

    try {
      const src = window.cv.imread(originalCanvas);
      const imgWidth = originalCanvas.width;
      const imgHeight = originalCanvas.height;

      // 依照畫布尺寸調整 polygon
      const scaledPolygon = predictionData.polygon.map((point) => [
        point[0] * (imgWidth / imageInfo.width),
        point[1] * (imgHeight / imageInfo.height),
      ]);

      const sortedPolygon = sortPolygonClockwise(scaledPolygon);
      if (sortedPolygon.length !== 4) {
        setError('Invalid number of points in polygon.');
        console.log('sortedPolygon is not 4 corners:', sortedPolygon);
        return;
      }

      // 重新設定攤平 canvas 大小，避免出現空白
      transformedCanvas.width = outputWidth;
      transformedCanvas.height = outputHeight;

      const srcPts = window.cv.matFromArray(4, 1, window.cv.CV_32FC2, sortedPolygon.flat());
      const dstPts = window.cv.matFromArray(
        4,
        1,
        window.cv.CV_32FC2,
        [0, 0, outputWidth, 0, outputWidth, outputHeight, 0, outputHeight]
      );

      const M = window.cv.getPerspectiveTransform(srcPts, dstPts);
      const dst = new window.cv.Mat();
      const dsize = new window.cv.Size(outputWidth, outputHeight);

      window.cv.warpPerspective(
        src,
        dst,
        M,
        dsize,
        window.cv.INTER_LINEAR,
        window.cv.BORDER_CONSTANT,
        new window.cv.Scalar()
      );
      window.cv.imshow(transformedCanvas, dst);

      src.delete();
      dst.delete();
      srcPts.delete();
      dstPts.delete();
      M.delete();

      console.log('Perspective transform done, canvas updated.');
    } catch (e) {
      console.error('Error performing perspective transform:', e);
      setError('Error performing perspective transform: ' + e.message);
    }
  };

  const currentFile = selectedFile;

  // -----------------------------------
  //                RENDER
  // -----------------------------------
  return (
    <div className={styles.demoWrapper}>
      <Space direction="vertical" style={{ width: '100%' }}>

        {/* OpenCV Download Section */}
        <Card title="OpenCV Download Status" style={{ marginBottom: 16 }}>
          {!openCvLoaded && (
            <Space direction="vertical">
              <Button
                type="primary"
                onClick={downloadOpenCv}
                loading={isDownloading}
              >
                {isDownloading ? 'Downloading...' : 'Retry Download OpenCV'}
              </Button>

              {/* 進度條 */}
              {(isDownloading || downloadProgress > 0) && (
                <Progress
                  percent={downloadProgress}
                  status={downloadError ? 'exception' : 'active'}
                />
              )}

              {/* 下載錯誤 */}
              {downloadError && (
                <Alert
                  message="OpenCV Download Error"
                  description={downloadError}
                  type="error"
                  showIcon
                />
              )}
            </Space>
          )}

          {openCvLoaded && (
            <Alert
              message="OpenCV.js loaded successfully."
              type="success"
              showIcon
            />
          )}
        </Card>

        {/* 上傳 / 狀態顯示 */}
        <Row justify="space-between" align="middle">
          <Col>
            <Upload
              beforeUpload={handleFileChange}
              showUploadList={false}
              disabled={apiStatus !== 'online'}
            >
              <Button
                icon={<UploadOutlined />}
                disabled={apiStatus !== 'online'}
              >
                {chooseFileLabel}
              </Button>
            </Upload>
          </Col>
          <Col>
            {apiStatus === 'online' && <Text type="success">🟢</Text>}
            {apiStatus === 'offline' && <Text type="danger">🔴</Text>}
            {apiStatus === null && <Text type="secondary">⚪</Text>}
          </Col>
        </Row>

        {imageInfo && currentFile && originalImageInfo && (
          <Card className={styles.demoCard} title={imageInfoTitle}>
            <ul className={styles.infoList}>
              <li>{fileNameLabel}: {currentFile.name}</li>
              <li>{fileSizeLabel}: {Math.round(currentFile.size / 1024)} KB</li>
              <li>{fileTypeLabel}: {currentFile.type}</li>
              <li>{imageSizeLabel}: {originalImageInfo.width} x {originalImageInfo.height} pixel</li>
            </ul>
          </Card>
        )}

        <Row gutter={16}>
          <Col span={12}>
            <canvas ref={originalCanvasRef} className={styles.demoCanvas}></canvas>
          </Col>
          <Col span={12}>
            <canvas ref={processedCanvasRef} className={styles.demoCanvas}></canvas>
          </Col>
        </Row>

        <Row justify="center" gutter={16}>
          <Col>
            <Button
              type="primary"
              onClick={uploadImage}
              disabled={apiStatus !== 'online'}
            >
              {uploadButtonLabel}
            </Button>
          </Col>
          <Col>
            {predictionData && !warning && (
              <Button icon={<DownloadOutlined />} onClick={downloadJSON}>
                {downloadButtonLabel}
              </Button>
            )}
          </Col>
          <Col>
            <Button icon={<ClearOutlined />} onClick={clearAll}>
              {clearButtonLabel}
            </Button>
          </Col>
        </Row>

        {isLoading && (
          <Spin tip={processingMessage} />
        )}

        {error && (
          <Alert message={error} type="error" showIcon />
        )}

        {warning && (
          <Alert message={warning} type="warning" showIcon />
        )}

        {imageInfo && (
          <>
            <Divider />
            <Title level={3}>{inferenceInfoTitle}</Title>
            <ul className={styles.infoList}>
              <li>{inferenceTimeLabel}: {inferenceTime.toFixed(2)} sec</li>
              <li>{timestampLabel}: {timestamp}</li>
            </ul>
          </>
        )}

        {predictionData && !warning && (
          <>
            <Divider />
            <Title level={3}>{polygonInfoTitle}</Title>
            <ul className={styles.infoList}>
              {predictionDataScale?.polygon?.map((point, index) => (
                <li key={index}>
                  Point {index + 1}: ({Math.round(point[0])}, {Math.round(point[1])})
                </li>
              ))}
            </ul>
          </>
        )}

        {predictionData && !warning && (
          <>
            <Divider />
            <Title level={3}>{TransformedTitle}</Title>
            <Space>
              <Text>{TransformedWidthLabel}:</Text>
              <InputNumber value={outputWidth} onChange={(val) => setOutputWidth(val)} />
              <Text>{TransformedHeightLabel}:</Text>
              <InputNumber value={outputHeight} onChange={(val) => setOutputHeight(val)} />
            </Space>
            <div className={styles.canvasWrapper}>
              <canvas ref={transformedCanvasRef} className={styles.demoCanvas}></canvas>
            </div>
            <div style={{ textAlign: 'center' }}>
              <Button icon={<DownloadOutlined />} onClick={downloadTransformedImage}>
                {TransformedButtonLabel}
              </Button>
            </div>
          </>
        )}
      </Space>
    </div>
  );
};

export default DocAlignerDemo;
