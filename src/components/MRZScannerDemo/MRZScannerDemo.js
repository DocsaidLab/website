import { ClearOutlined, DownloadOutlined, UploadOutlined } from '@ant-design/icons';
import {
  Alert,
  Button,
  Card,
  Col,
  List,
  Row,
  Space,
  Spin,
  Switch,
  Typography,
  Upload
} from 'antd';
import React, { useEffect, useRef, useState } from 'react';

import { useApiStatus } from '../Common/hooks/useApiStatus';
import { useImageUploader } from '../Common/hooks/useImageUploader';
import { downloadJSON } from '../Common/utils/fileUtils';
import { drawPolygon, drawPolygonSimple } from '../Common/utils/imageUtils';

import styles from './styles.module.css';

const { Title, Text } = Typography;

const MRZScannerDemo = ({
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
  mrzTextsTitle,
  inferenceTimeLabel,
  timestampLabel,
  fileNameLabel,
  fileSizeLabel,
  fileTypeLabel,
  imageSizeLabel,
  mrzOptionsTitle,
  doDocAlignLabel,
  doCenterCropLabel,
  doPostprocessLabel,
  externalImage
}) => {
  const { apiStatus } = useApiStatus('https://api.docsaid.org/mrzscanner-predict');

  // Canvas 參照
  const originalCanvasRef = useRef(null);
  const processedCanvasRef = useRef(null);

  // 上傳 hook
  const {
    selectedFile,
    imageInfo,
    originalImageInfo,
    warning,
    error,
    setError,
    setWarning,
    handleFileChange,
    handleExternalImageChange,
    clearAll: clearUploader
  } = useImageUploader({
    originalCanvasRef,
    warningMessage
  });

  // 狀態
  const [isLoading, setIsLoading] = useState(false);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [timestamp, setTimestamp] = useState(0);
  const [docPolygon, setDocPolygon] = useState(null);
  const [mrzPolygon, setMrzPolygon] = useState(null);
  const [mrzTexts, setMrzTexts] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');

  // 參數 (logging 移除，預設true)
  const [doDocAlign, setDoDocAlign] = useState(true);        // 預設 true
  const [doCenterCrop, setDoCenterCrop] = useState(true);    // 預設 true
  const [doPostprocess, setDoPostprocess] = useState(false); // 預設 false

  // 如果有 externalImage -> 直接載入
  useEffect(() => {
    if (externalImage) {
      handleExternalImageChange(externalImage);
    }
  }, [externalImage]);

  // 清除
  const clearAll = () => {
    clearUploader();
    setInferenceTime(0);
    setTimestamp(0);
    setDocPolygon(null);
    setMrzPolygon(null);
    setMrzTexts([]);
    setErrorMsg('');

    if (processedCanvasRef.current) {
      const ctx = processedCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, processedCanvasRef.current.width, processedCanvasRef.current.height);
    }
  };

  // 上傳 & 呼叫 API
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
    setDocPolygon(null);
    setMrzPolygon(null);
    setMrzTexts([]);
    setErrorMsg('');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('do_doc_align', doDocAlign);
    formData.append('do_center_crop', doCenterCrop);
    formData.append('do_postprocess', doPostprocess);
    formData.append('logging', true);

    fetch('https://api.docsaid.org/mrzscanner-predict', {
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
        setInferenceTime(data.inference_time || 0);
        setTimestamp(data.timestamp || 0);
        setErrorMsg(data.error_msg || '');

        const dp = data.doc_polygon;
        const mp = data.mrz_polygon;
        const texts = data.mrz_texts;

        if (dp && dp.length > 0) setDocPolygon(dp);
        if (mp && mp.length > 0) setMrzPolygon(mp);
        if (texts && texts.length > 0) setMrzTexts(texts);

        // 繪製
        processedCanvas.width = originalCanvas.width;
        processedCanvas.height = originalCanvas.height;
        const ctx = processedCanvas.getContext('2d');
        ctx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
        ctx.drawImage(originalCanvas, 0, 0);

        // doc_polygon -> 多色+箭頭
        if (dp && dp.length > 0) {
          drawPolygon(ctx, dp);
        }
        // mrz_polygon -> 單一綠色 + 排序
        if (mp && mp.length > 0) {
          drawPolygonSimple(ctx, mp, '#00FF00');
        }
      })
      .catch((err) => {
        console.error('Error:', err);
        setError(`${errorMessage.uploadError}: ${err.message || err}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
  };

  // 下載 JSON
  const handleDownloadJSON = () => {
    const resultData = {
      doc_polygon: docPolygon,
      mrz_polygon: mrzPolygon,
      mrz_texts: mrzTexts,
      inference_time: inferenceTime,
      timestamp,
      error_msg: errorMsg
    };
    downloadJSON(resultData, 'mrzscanner_result.json');
  };

  const currentFile = selectedFile;

  return (
    <div className={styles.demoWrapper}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">

        {/* 上傳 / 狀態 */}
        <Row justify="space-between" align="middle">
          <Col>
            <Upload
              beforeUpload={handleFileChange}
              showUploadList={false}
              disabled={apiStatus !== 'online'}
            >
              <Button icon={<UploadOutlined />} disabled={apiStatus !== 'online'}>
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

        {/* 圖像資訊：維持原本 ul */}
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

        {/* MRZ Scanner Options */}
        <Card className={styles.demoCard} title={mrzOptionsTitle}>
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Space>
                <Switch checked={doDocAlign} onChange={val => setDoDocAlign(val)} />
                <Text>{doDocAlignLabel}</Text>
              </Space>
            </Col>
            <Col xs={24} sm={8}>
              <Space>
                <Switch checked={doCenterCrop} onChange={val => setDoCenterCrop(val)} />
                <Text>{doCenterCropLabel}</Text>
              </Space>
            </Col>
            <Col xs={24} sm={8}>
              <Space>
                <Switch checked={doPostprocess} onChange={val => setDoPostprocess(val)} />
                <Text>{doPostprocessLabel}</Text>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* Canvas */}
        <Row gutter={16}>
          <Col span={12}>
            <canvas ref={originalCanvasRef} className={styles.demoCanvas}></canvas>
          </Col>
          <Col span={12}>
            <canvas ref={processedCanvasRef} className={styles.demoCanvas}></canvas>
          </Col>
        </Row>

        {/* 按鈕 */}
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
            {(docPolygon || mrzPolygon || mrzTexts.length > 0) && (
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

        {/* Loading */}
        {isLoading && <Spin tip={processingMessage} />}

        {/* 前端錯誤、警告 */}
        {error && <Alert message={error} type="error" showIcon />}
        {warning && <Alert message={warning} type="warning" showIcon />}

        {/* MRZ 文字：若有才顯示 */}
        {mrzTexts && mrzTexts.length > 0 && (
          <Card
            className={styles.demoCard}
            title={mrzTextsTitle}
          >
            <Space direction="vertical" style={{ width: '100%' }}>
              {mrzTexts.map((txt, i) => (
                <Text
                  key={i}
                  style={{
                    fontSize: '1.1rem',
                    fontFamily: 'Monospace',
                    background: '#fdfdfd',
                    display: 'inline-block',
                    padding: '4px 8px',
                    borderRadius: '4px',
                    marginBottom: '4px'
                  }}
                >
                  {txt}
                </Text>
              ))}
            </Space>

            {/* 後端回傳錯誤訊息 */}
            {errorMsg && errorMsg !== 'ErrorCodes.NO_ERROR' && (
              <Alert
                style={{ marginTop: 16 }}
                message="Backend Error Message"
                description={errorMsg}
                type="error"
                showIcon
              />
            )}
            {errorMsg === 'ErrorCodes.NO_ERROR' && (
              <Alert
                style={{ marginTop: 16 }}
                message="Backend Status"
                description="No error reported by the backend."
                type="success"
                showIcon
              />
            )}
          </Card>
        )}

        {/* 偵測結果 */}
        {(docPolygon || mrzPolygon) && (
          <Card className={styles.demoCard} title={polygonInfoTitle} size="small">
            {docPolygon && (
              <div style={{ marginBottom: 12 }}>
                <Text strong style={{ color: 'blue' }}>
                  doc_polygon (Color + arrows):
                </Text>
                <List
                  size="small"
                  dataSource={docPolygon}
                  renderItem={(pt, idx) => (
                    <List.Item key={idx}>
                      ({Math.round(pt[0])}, {Math.round(pt[1])})
                    </List.Item>
                  )}
                />
              </div>
            )}
            {mrzPolygon && (
              <div>
                <Text strong style={{ color: 'green' }}>
                  mrz_polygon (Green):
                </Text>
                <List
                  size="small"
                  dataSource={mrzPolygon}
                  renderItem={(pt, idx) => (
                    <List.Item key={idx}>
                      ({Math.round(pt[0])}, {Math.round(pt[1])})
                    </List.Item>
                  )}
                />
              </div>
            )}
          </Card>
        )}

        {/* 推論時間卡片：避免顯示 0，只有在 inferenceTime>0 或 timestamp>0 才顯示 */}
        {(inferenceTime > 0 || timestamp > 0) && (
          <Card className={styles.demoCard} title={inferenceInfoTitle} size="small">
            <Row>
              <Col span={12}>
                <Text strong>{inferenceTimeLabel}：</Text>
                <Text>{inferenceTime.toFixed(2)} sec</Text>
              </Col>
              <Col span={12}>
                <Text strong>{timestampLabel}：</Text>
                <Text>{timestamp}</Text>
              </Col>
            </Row>
          </Card>
        )}

      </Space>
    </div>
  );
};

export default MRZScannerDemo;
