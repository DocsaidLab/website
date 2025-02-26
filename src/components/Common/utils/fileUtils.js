
/**
 * 驗證檔案格式
 * @param {File} file - 上傳的檔案
 * @param {Array<string>} validTypes - 可接受的檔案 MIME type
 * @returns {boolean}
 */
export function validateFileType(file, validTypes = ['image/jpeg', 'image/png', 'image/webp']) {
    return validTypes.includes(file.type);
  }

  /**
   * 將物件或 JSON 資料下載為 .json 檔案
   * @param {Object} data - 要下載的 JSON 物件
   * @param {string} fileName - 下載時的預設檔名
   */
  export function downloadJSON(data, fileName = 'prediction.json') {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute('href', dataStr);
    downloadAnchorNode.setAttribute('download', fileName);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  }
