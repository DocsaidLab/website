import React, { useEffect, useRef, useState } from 'react';
import '../css/DocAlignerDemo.css';

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
  externalImage,
}) => {
  const fileInputRef = useRef(null);
  const originalCanvasRef = useRef(null);
  const processedCanvasRef = useRef(null);
  const [predictionData, setPredictionData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [warning, setWarning] = useState(null);
  const [imageInfo, setImageInfo] = useState(null);
  const [originalImageInfo, setOriginalImageInfo] = useState(null);
  const [scale, setScale] = useState(1);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [timestamp, setTimestamp] = useState(0);
  const [selectedFile, setSelectedFile] = useState(null);

  useEffect(() => {
    if (externalImage) {
      handleExternalImageChange(externalImage);
    }
  }, [externalImage]);

  const handleExternalImageChange = (imageSource) => {
    const canvas = originalCanvasRef.current;
    clearAll();

    setError(null);
    setWarning(null);
    setPredictionData(null);
    setImageInfo(null);

    const img = new Image();
    img.crossOrigin = "Anonymous"; // Avoid CORS issues
    img.onload = function () {
      const originalWidth = img.width;
      const originalHeight = img.height;
      let scale = 1;
      if (img.width > 1000 || img.height > 1000) {
        scale = 1000 / Math.max(img.width, img.height);
      }
      setScale(scale);

      const scaledWidth = img.width * scale;
      const scaledHeight = img.height * scale;

      canvas.width = scaledWidth;
      canvas.height = scaledHeight;

      if (img.width > 4000 || img.height > 4000) {
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

  const handleFileChange = () => {
    const fileInput = fileInputRef.current;
    const canvas = originalCanvasRef.current;

    clearAll();

    if (!fileInput.files || fileInput.files.length === 0) {
      setError(errorMessage.chooseFile);
      clearAll();
      return;
    }

    const file = fileInput.files[0];

    if (!validateFileType(file)) {
      setError(errorMessage.invalidFileType);
      clearAll();
      return;
    }

    setError(null);
    setWarning(null);
    setPredictionData(null);
    setImageInfo(null);

    const reader = new FileReader();
    reader.onload = function(event) {
      const img = new Image();
      img.onload = function() {
        const originalWidth = img.width;
        const originalHeight = img.height;
        let scale = 1;
        if (img.width > 1000 || img.height > 1000) {
          scale = 1000 / Math.max(img.width, img.height);
        }
        setScale(scale);

        const scaledWidth = img.width * scale;
        const scaledHeight = img.height * scale;

        canvas.width = scaledWidth;
        canvas.height = scaledHeight;

        if (img.width > 4000 || img.height > 4000) {
          setWarning(warningMessage.imageTooLarge);
        }

        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);

        adjustCanvasSize(canvas, scaledWidth, scaledHeight);
        setImageInfo({ width: scaledWidth, height: scaledHeight });
        setOriginalImageInfo({ width: originalWidth, height: originalHeight });

        // Convert canvas to Blob and create a File object
        canvas.toBlob(function(blob) {
          const scaledFile = new File([blob], file.name, { type: file.type });
          setSelectedFile(scaledFile);
        }, file.type);
      };
      img.src = event.target.result;
    };

    reader.readAsDataURL(file);
  };

  const clearAll = () => {
    setPredictionData(null);
    setWarning(null);
    setImageInfo(null);
    setOriginalImageInfo(null);
    setSelectedFile(null);
    setInferenceTime(0);
    setScale(1);
    const originalCtx = originalCanvasRef.current.getContext('2d');
    const processedCtx = processedCanvasRef.current.getContext('2d');
    originalCtx.clearRect(0, 0, originalCanvasRef.current.width, originalCanvasRef.current.height);
    processedCtx.clearRect(0, 0, processedCanvasRef.current.width, processedCanvasRef.current.height);
  };

  const uploadImage = () => {
    const originalCanvas = originalCanvasRef.current;
    const processedCanvas = processedCanvasRef.current;

    let file = null;

    if (selectedFile) {
      file = selectedFile;
    } else {
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

        // Adjust polygon coordinates back to original image dimensions
        const adjustedPolygon = data.polygon.map(point => [
          point[0] / scale,
          point[1] / scale,
        ]);

        // Update prediction data with adjusted polygon
        setPredictionData({ ...data, polygon: adjustedPolygon });

        const ctx = processedCanvas.getContext('2d');
        ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
        processedCanvas.width = originalCanvas.width;
        processedCanvas.height = originalCanvas.height;
        ctx.drawImage(originalCanvas, 0, 0);

        adjustCanvasSize(processedCanvas, originalCanvas.width, originalCanvas.height);
        drawPolygon(ctx, data.polygon); // Draw using scaled coordinates
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
    const containerWidth = document.querySelector('.images-container').clientWidth;
    const maxWidth = containerWidth / 2 - 20;

    if (width > maxWidth) {
      const aspectRatio = height / width;
      canvas.style.width = `${maxWidth}px`;
      canvas.style.height = `${maxWidth * aspectRatio}px`;
    } else {
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
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
    // First, sort points by y-coordinate (ascending)
    const sortedByY = polygon.slice().sort((a, b) => a[1] - b[1]);

    // Top two points (smallest y)
    const topPoints = sortedByY.slice(0, 2);

    // Bottom two points (largest y)
    const bottomPoints = sortedByY.slice(2);

    // Sort top points by x-coordinate to get left and right
    const [topLeft, topRight] = topPoints.sort((a, b) => a[0] - b[0]);

    // Sort bottom points by x-coordinate to get left and right
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

  const currentFile = selectedFile || (fileInputRef.current && fileInputRef.current.files[0]);

  return (
    <div className="doc-aligner-demo">

      <div>
        <label className="custom-file-upload">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
          />
          {chooseFileLabel}
        </label>
      </div>

      <br />

      {imageInfo && currentFile && originalImageInfo && (
        <div id="imageInfo">
          <h3>{imageInfoTitle}</h3>
          <ul>
            <li>{fileNameLabel}: {currentFile.name}</li>
            <li>{fileSizeLabel}: {Math.round(currentFile.size / 1024)} KB</li>
            <li>{fileTypeLabel}: {currentFile.type}</li>
            <li>{imageSizeLabel}: {originalImageInfo.width} x {originalImageInfo.height} pixel</li>
          </ul>
        </div>
      )}

      <div className="images-container">
        <canvas ref={originalCanvasRef}></canvas>
        <canvas ref={processedCanvasRef}></canvas>
      </div>

      <div align="center">
        <button id="uploadButton" onClick={uploadImage}>{uploadButtonLabel}</button>
        {predictionData && !warning && (
          <button id="downloadButton" onClick={downloadJSON}>
            {downloadButtonLabel}
          </button>
        )}
        <button id="clearButton" onClick={clearAll}>{clearButtonLabel}</button>
      </div>
      {isLoading && <div id="loadingIndicator">{processingMessage}</div>}
      {error && <div id="errorMessage">{error}</div>}
      {warning && <div id="warningMessage">{warning}</div>}

      <hr />

      {imageInfo && (
        <div id="imageInfo">
          <h3>{inferenceInfoTitle}</h3>
          <ul>
            <li>{inferenceTimeLabel}: {inferenceTime.toFixed(2)} sec </li>
            <li>{timestampLabel}: {timestamp}</li>
          </ul>
        </div>
      )}

      <hr />

      {predictionData && !warning && (
        <div id="polygonInfo">
          <h3>{polygonInfoTitle}</h3>
          <ul>
            {predictionData.polygon.map((point, index) => (
              <li key={index}>
                Point {index + 1}：({Math.round(point[0])}, {Math.round(point[1])})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default DocAlignerDemo;
