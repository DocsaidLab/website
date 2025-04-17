// src/hooks/useLayoutType.js
import { Grid } from 'antd';
const { useBreakpoint } = Grid;

/**
 * 根據 antd 提供的斷點資訊，回傳 'threeCards' | 'twoCards' | 'oneCard'
 */
export default function useLayoutType() {
  const screens = useBreakpoint();

  // 修正：若要在 576px 就顯示兩欄，改用 screens.sm
  if (screens.lg) return 'threeCards';
  if (screens.sm) return 'twoCards';
  return 'oneCard';
}
