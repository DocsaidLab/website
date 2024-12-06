import { ClearOutlined, DownloadOutlined, UploadOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  Divider,
  InputNumber,
  Row,
  Space,
  Spin,
  Typography,
  Upload
} from 'antd';
import React, { useEffect, useRef, useState } from 'react';
import '../css/DocAlignerDemo.css';

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
  const [openCvLoaded, setOpenCvLoaded] = useState(false);
  const [outputWidth, setOutputWidth] = useState(768);
  const [outputHeight, setOutputHeight] = useState(480);
  const [apiStatus, setApiStatus] = useState(null);

  useEffect(() => {
    if (externalImage) {
      handleExternalImageChange(externalImage);
    }
  }, [externalImage]);

  useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await fetch('https://api.docsaid.org/docaligner-predict');
      if (response.ok) {
        setApiStatus('online'); // API 暢通
      } else {
        setApiStatus('offline'); // API 故障
      }
    } catch (error) {
      setApiStatus('offline'); // 請求失敗，視為 API 故障
      console.error('Error checking API status:', error);
    }
  };

  // Load OpenCV.js
  useEffect(() => {
    window.Module = {
      onRuntimeInitialized() {
        console.log('OpenCV.js is ready');
        setOpenCvLoaded(true);
      },
    };
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.x/opencv.js';
    script.async = true;
    script.onload = () => {
      console.log('OpenCV.js script loaded');
    };
    script.onerror = () => {
      console.error('Failed to load OpenCV.js');
      setError('Failed to load OpenCV.js');
    };
    document.body.appendChild(script);

    return () => {
      document.body.removeChild(script);
    };
  }, []);

  useEffect(() => {
    if (predictionData && openCvLoaded) {
      performPerspectiveTransform();
    }
  }, [predictionData, openCvLoaded, outputWidth, outputHeight]);

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
    img.crossOrigin = "Anonymous";
    img.onload = function () {
      const originalWidth = img.width;
      const originalHeight = img.height;
      let scale = 1;
      if (img.width > 2000 || img.height > 2000) {
        scale = 2000 / Math.max(img.width, img.height);
      }
      setScale(scale);

      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;

      canvas.width = scaledWidth;
      canvas.height = scaledHeight;

      if (img.width > 5000 || img.height > 5000) {
        setWarning(warningMessage.imageTooLarge);
      }

      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

      adjustCanvasSize(canvas, scaledWidth, scaledHeight);
      setImageInfo({ width: scaledWidth, height: scaledHeight });
      setOriginalImageInfo({ width: originalWidth, height: originalHeight });

      // Convert canvas to Blob and create a File object
      canvas.toBlob(function (blob) {
        const file = new File([blob], "example.jpg", { type: "image/jpeg" });
        setSelectedFile(file);
      }, "image/jpeg");
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
        let scale = 1;
        if (img.width > 2000 || img.height > 2000) {
          scale = 2000 / Math.max(img.width, img.height);
        }
        setScale(scale);

        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;

        canvas.width = scaledWidth;
        canvas.height = scaledHeight;

        if (img.width > 5000 || img.height > 5000) {
          setWarning(warningMessage.imageTooLarge);
        }

        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

        adjustCanvasSize(canvas, scaledWidth, scaledHeight);
        setImageInfo({ width: scaledWidth, height: scaledHeight });
        setOriginalImageInfo({ width: originalWidth, height: originalHeight });

        // Convert canvas to Blob and create a File object
        canvas.toBlob(function (blob) {
          const scaledFile = new File([blob], file.name, { type: file.type });
          setSelectedFile(scaledFile);
        }, file.type);
      };
      img.src = event.target.result;
    };

    reader.readAsDataURL(file);
    return false; // 禁止 Upload 自動上傳行為
  };

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
    .then(response => {
      if (!response.ok) {
        throw new Error(errorMessage.networkError);
      }
      return response.json();
    })
    .then(data => {
      if (!data.polygon || data.polygon.length === 0) {
        setWarning(warningMessage.noPolygon);
        setIsLoading(false);
        return;
      }

      setInferenceTime(data.inference_time);
      setTimestamp(data.timestamp);
      setPredictionData(data);

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

      adjustCanvasSize(processedCanvas, originalCanvas.width, originalCanvas.height);
      drawPolygon(ctx, data.polygon);
    })
    .catch(error => {
      console.error('Error:', error);
      setError(`${errorMessage.uploadError}: ${error.message || error}`);
    })
    .finally(() => {
      setIsLoading(false);
    });
  };

  const adjustCanvasSize = (canvas, width, height) => {
    // 使用一般方式控制 canvas 大小即可，不一定需要動態調整
    // 此處略過以保留簡單性
  };

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

      ctx.beginPath();
      ctx.arc(p1[0], p1[1], thickness * 2, 0, Math.PI * 2, false);
      ctx.fillStyle = color;
      ctx.fill();

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

  const downloadJSON = () => {
    if (!predictionData) return;
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(predictionData));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "prediction.json");
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

  const performPerspectiveTransform = () => {
    if (!predictionData || !openCvLoaded || !cv || !cv.imread) return;

    const originalCanvas = originalCanvasRef.current;
    const transformedCanvas = transformedCanvasRef.current;

    if (!originalCanvas || !transformedCanvas) return;

    try {
      const src = cv.imread(originalCanvas);
      const imgWidth = originalCanvas.width;
      const imgHeight = originalCanvas.height;
      const scaledPolygon = predictionData.polygon.map(point => [
        point[0] * (imgWidth / imageInfo.width),
        point[1] * (imgHeight / imageInfo.height),
      ]);

      const sortedPolygon = sortPolygonClockwise(scaledPolygon);

      if (sortedPolygon.length !== 4) {
        setError('Invalid number of points in polygon.');
        return;
      }

      const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, sortedPolygon.flat());
      const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0,
        outputWidth, 0,
        outputWidth, outputHeight,
        0, outputHeight
      ]);

      const M = cv.getPerspectiveTransform(srcPts, dstPts);
      const dst = new cv.Mat();
      const dsize = new cv.Size(outputWidth, outputHeight);

      cv.warpPerspective(src, dst, M, dsize, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
      cv.imshow(transformedCanvas, dst);

      src.delete();
      dst.delete();
      srcPts.delete();
      dstPts.delete();
      M.delete();
    } catch (e) {
      console.error('Error performing perspective transform:', e);
      setError('Error performing perspective transform: ' + e.message);
    }
  };

  const currentFile = selectedFile;

  return (
    <div style={{ padding: '20px' }}>
      <Space direction="vertical" style={{ width: '100%' }}>

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
          <Card title={imageInfoTitle}>
            <ul>
              <li>{fileNameLabel}: {currentFile.name}</li>
              <li>{fileSizeLabel}: {Math.round(currentFile.size / 1024)} KB</li>
              <li>{fileTypeLabel}: {currentFile.type}</li>
              <li>{imageSizeLabel}: {originalImageInfo.width} x {originalImageInfo.height} pixel</li>
            </ul>
          </Card>
        )}

        <Row gutter={16}>
          <Col span={12}>
            <canvas ref={originalCanvasRef} style={{ width: '100%', border: '1px solid #ccc' }}></canvas>
          </Col>
          <Col span={12}>
            <canvas ref={processedCanvasRef} style={{ width: '100%', border: '1px solid #ccc' }}></canvas>
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
            <ul>
              <li>{inferenceTimeLabel}: {inferenceTime.toFixed(2)} sec </li>
              <li>{timestampLabel}: {timestamp}</li>
            </ul>
          </>
        )}

        {predictionData && !warning && (
          <>
            <Divider />
            <Title level={3}>{polygonInfoTitle}</Title>
            <ul>
              {predictionDataScale.polygon.map((point, index) => (
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
              <InputNumber value={outputWidth} onChange={(val)=>setOutputWidth(val)} />
              <Text>{TransformedHeightLabel}:</Text>
              <InputNumber value={outputHeight} onChange={(val)=>setOutputHeight(val)} />
            </Space>
            <div style={{ marginTop: '20px', textAlign: 'center' }}>
              <canvas ref={transformedCanvasRef} style={{ border: '1px solid #ccc', maxWidth: '100%' }}></canvas>
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
