/**
 * 說明：專門放置「純函式」的圖像工具方法，如多邊形排序、繪製函式等
 */

/**
 * 將多邊形頂點按照「左上、右上、右下、左下」順序排序
 * @param {Array<[number, number]>} polygon - 例如 [[100,150],[400,150],[400,300],[100,300]]
 * @returns {Array<[number, number]>}
 */
export function sortPolygonClockwise(polygon) {
    const sortedByY = polygon.slice().sort((a, b) => a[1] - b[1]);
    const topPoints = sortedByY.slice(0, 2).sort((a, b) => a[0] - b[0]);
    const bottomPoints = sortedByY.slice(2).sort((a, b) => a[0] - b[0]);
    // [左上, 右上, 右下, 左下]
    return [topPoints[0], topPoints[1], bottomPoints[1], bottomPoints[0]];
  }

  /**
   * 在 Canvas 上畫箭頭
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} fromX
   * @param {number} fromY
   * @param {number} toX
   * @param {number} toY
   * @param {number} thickness - 線條粗細
   * @param {string} color - 顏色 ('rgb(...)' 或其他CSS顏色)
   */
  export function drawArrow(ctx, fromX, fromY, toX, toY, thickness, color) {
    const headlen = thickness * 5;
    const angle = Math.atan2(toY - fromY, toX - fromX);

    // 畫主線
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;
    ctx.stroke();

    // 畫箭頭
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
  }

  /**
   * 在 Canvas 上畫多邊形（並繪製每個頂點及箭頭）
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<[number, number]>} polygon - [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
   */
  export function drawPolygon(ctx, polygon) {
    const colors = [
      [0, 255, 255],
      [0, 255, 0],
      [255, 0, 0],
      [255, 255, 0]
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
  }

  function sortPolygonByCenterAngle(polygon) {
    if (!polygon || polygon.length < 2) return polygon || [];
    // 計算多邊形頂點中心
    let cx = 0, cy = 0;
    polygon.forEach(pt => {
      cx += pt[0];
      cy += pt[1];
    });
    cx /= polygon.length;
    cy /= polygon.length;

    // 按與中心點的角度排序
    return polygon.slice().sort((a, b) => {
      const angleA = Math.atan2(a[1] - cy, a[0] - cx);
      const angleB = Math.atan2(b[1] - cy, b[0] - cx);
      return angleA - angleB;
    });
  }

  /**
   * 使用單一顏色繪製多邊形，先行以中心角度排序以避免交叉
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<[number, number]>} polygon
   * @param {string} color
   */
  export function drawPolygonSimple(ctx, polygon, color = '#00FF00') {
    const sorted = sortPolygonByCenterAngle(polygon);
    if (sorted.length < 2) return;

    const thickness = Math.max(ctx.canvas.width * 0.005, 1);
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = thickness;

    ctx.moveTo(sorted[0][0], sorted[0][1]);
    for (let i = 1; i < sorted.length; i++) {
      ctx.lineTo(sorted[i][0], sorted[i][1]);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.restore();
  }