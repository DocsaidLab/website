// src/hooks/useLayoutType.js
import { Grid } from 'antd';
const { useBreakpoint } = Grid;

/**
 * 傳入參數來決定是否開啟 3 欄顯示
 */
export default function useLayoutType(options = {}) {
  const { maxColumns = 3 } = options;
  const screens = useBreakpoint();

  // 如果 maxColumns < 3，就自動不會回傳 threeCards
  if (screens.lg && maxColumns >= 3) return 'threeCards';
  if (screens.sm && maxColumns >= 2) return 'twoCards';
  return 'oneCard';
}
