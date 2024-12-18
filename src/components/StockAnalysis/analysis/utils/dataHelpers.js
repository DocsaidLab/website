import { mean, std } from 'mathjs';

export function findMax(data, field) {
  if (!Array.isArray(data) || data.length === 0) return null;
  return data.reduce((prev, curr) => (curr[field] > prev[field] ? curr : prev));
}

export function findMin(data, field) {
  if (!Array.isArray(data) || data.length === 0) return null;
  return data.reduce((prev, curr) => (curr[field] < prev[field] ? curr : prev));
}

export function formatNumber(num) {
  if (typeof num !== 'number' || isNaN(num)) return 'N/A';
  return num.toFixed(2);
}

export { mean, std };

export function formatPrice(value) {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A'; // 或給一個預設值
  }

  if (value >= 1000) {
    return value.toFixed(0);
  } else if (value >= 500) {
    return value.toFixed(0);
  } else if (value >= 100) {
    return value.toFixed(1);
  } else if (value >= 50) {
    return value.toFixed(1);
  } else if (value >= 10) {
    return value.toFixed(2);
  } else if (value >= 5) {
    return value.toFixed(2);
  } else {
    return value.toFixed(2);
  }
}
