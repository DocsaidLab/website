// src/hooks/useLayoutType.js
import { Grid } from 'antd';
const { useBreakpoint } = Grid;

/**
 * 根據 antd 提供的斷點資訊，回傳 'threeCards' | 'twoCards' | 'oneCard'
 */
export default function useLayoutType() {
  const screens = useBreakpoint();
  if (screens.lg) return 'threeCards';
  if (screens.md) return 'twoCards';
  return 'oneCard';
}
