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

import { useApiStatus } from '../Common/hooks/useApiStatus';
import { useImageUploader } from '../Common/hooks/useImageUploader';
import { useOpenCV } from '../Common/hooks/useOpenCV';
import { downloadJSON } from '../Common/utils/fileUtils';
import { drawPolygon, sortPolygonClockwise } from '../Common/utils/imageUtils';

import styles from './styles.module.css';

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
  // -------------------
  // (1) Hooks: OpenCV / API 狀態
  // -------------------
  const { apiStatus } = useApiStatus('https://api.docsaid.org/docaligner-predict');
  const {
    openCvLoaded,
    isDownloading,
    downloadProgress,
    downloadError,
    downloadOpenCv
  } = useOpenCV();

  // -------------------
  // (2) Canvas Ref
  // -------------------
  const originalCanvasRef = useRef(null);
  const processedCanvasRef = useRef(null);
  const transformedCanvasRef = useRef(null);

  // -------------------
  // (3) 用 useImageUploader 管理「上傳 / 載入圖片」相關狀態
  // -------------------
  const {
    selectedFile,
    scale,
    imageInfo,
    originalImageInfo,
    warning,
    error,
    setError,            // 方便在後面也可 setError
    setWarning,          // 方便在後面也可 setWarning
    handleFileChange,
    handleExternalImageChange,
    clearAll: clearUploader
  } = useImageUploader({
    originalCanvasRef,
    warningMessage
  });

  // 注意：我們把「清除畫面」的動作從先前的 `clearAll` 移到 hook 內的 `clearAll: clearUploader`。
  // 所以若元件需要做更多事一起清除，可在這裡包一層(如下)

  // -------------------
  //  (4) 其他狀態 (推理結果、loading等)
  // -------------------
  const [predictionData, setPredictionData] = useState(null);
  const [predictionDataScale, setPredictionDataScale] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [timestamp, setTimestamp] = useState(0);
  const [outputWidth, setOutputWidth] = useState(768);
  const [outputHeight, setOutputHeight] = useState(480);

  /**
   * 改寫：clearAll
   * 除了清除圖片，還要清除 predictionData / predictionDataScale / ... 之類
   */
  const clearAll = () => {
    // 先清除圖片 (hook 內會清理 originalCanvas)
    clearUploader();

    // 再清除預測
    setPredictionData(null);
    setPredictionDataScale(null);
    setInferenceTime(0);
    setTimestamp(0);
    setOutputWidth(768);
    setOutputHeight(480);

    // 清除 processedCanvas
    if (processedCanvasRef.current) {
      const ctx = processedCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, processedCanvasRef.current.width, processedCanvasRef.current.height);
    }
    // 清除 transformedCanvas
    if (transformedCanvasRef.current) {
      const ctx = transformedCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, transformedCanvasRef.current.width, transformedCanvasRef.current.height);
    }
  };

  // 若外部傳入影像路徑 -> 直接呼叫 Hook 的 handleExternalImageChange
  useEffect(() => {
    if (externalImage) {
      handleExternalImageChange(externalImage);
    }
  }, [externalImage]);

  // 頁面載入後，自動嘗試下載 OpenCV
  useEffect(() => {
    if (!openCvLoaded) {
      downloadOpenCv();
    }
    // eslint-disable-next-line
  }, []);

  // 預測完成且 OpenCV 就緒後 -> 進行透視轉換
  useEffect(() => {
    if (predictionData && openCvLoaded) {
      performPerspectiveTransform();
    }
  }, [predictionData, openCvLoaded, outputWidth, outputHeight]);

  // -------------------
  // (5) 上傳影像到後端
  // -------------------
  const uploadImage = () => {
    if (!selectedFile) {
      setError(errorMessage.chooseFile);
      clearAll();
      return;
    }

    const originalCanvas = originalCanvasRef.current;
    const processedCanvas = processedCanvasRef.current;
    if (!originalCanvas || !processedCanvas) return;

    setIsLoading(true);
    setError(null);
    setWarning(null);
    setPredictionData(null);
    setPredictionDataScale(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

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

        // 伺服器回傳 polygon -> 再放大成「原始 canvas」坐標
        const adjustedPolygon = data.polygon.map((point) => [
          point[0] / scale,
          point[1] / scale,
        ]);
        setPredictionDataScale({ ...data, polygon: adjustedPolygon });

        // 在 processedCanvas 畫圖
        processedCanvas.width = originalCanvas.width;
        processedCanvas.height = originalCanvas.height;
        const ctx = processedCanvas.getContext('2d');
        ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
        ctx.drawImage(originalCanvas, 0, 0);

        // 呼叫 from imageUtils
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

  // -------------------
  // (6) 下載 / 透視轉換
  // -------------------
  const handleDownloadJSON = () => {
    if (!predictionData) return;
    downloadJSON(predictionData, 'prediction.json');
  };

  const downloadTransformedImage = () => {
    if (!transformedCanvasRef.current) return;
    const dataURL = transformedCanvasRef.current.toDataURL('image/jpeg');
    const link = document.createElement('a');
    link.href = dataURL;
    link.download = 'transformed_image.jpg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const performPerspectiveTransform = () => {
    if (!predictionData || !openCvLoaded || !window.cv || !window.cv.imread) {
      console.log('Cannot transform yet: missing data or OpenCV not loaded.');
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

  // -------------------
  // Render
  // -------------------
  const currentFile = selectedFile;

  return (
    <div className={styles.demoWrapper}>
      <Space direction="vertical" style={{ width: '100%' }}>

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
              {(isDownloading || downloadProgress > 0) && (
                <Progress
                  percent={downloadProgress}
                  status={downloadError ? 'exception' : 'active'}
                />
              )}
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
              <Button icon={<DownloadOutlined />} onClick={handleDownloadJSON}>
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

        {/* 來自 hook 的 error / warning */}
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
