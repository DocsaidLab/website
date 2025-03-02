// src/theme/Navbar/Content/index.js
import { UserOutlined } from '@ant-design/icons';
import Link from "@docusaurus/Link";
import { Avatar, Button, Dropdown, Menu, message } from 'antd';
import React, { useState } from 'react';
import { useAuth } from '../../../context/AuthContext';

import { ErrorCauseBoundary, useThemeConfig } from '@docusaurus/theme-common';
import {
  splitNavbarItems,
  useNavbarMobileSidebar,
} from '@docusaurus/theme-common/internal';

import NavbarColorModeToggle from '@theme/Navbar/ColorModeToggle';
import NavbarLogo from '@theme/Navbar/Logo';
import NavbarMobileSidebarToggle from '@theme/Navbar/MobileSidebar/Toggle';
import NavbarSearch from '@theme/Navbar/Search';
import NavbarItem from '@theme/NavbarItem';
import SearchBar from '@theme/SearchBar';
import styles from './styles.module.css';

import AuthModal from '../../../components/AuthModal'; // 自訂的 Modal

function useNavbarItems() {
  return useThemeConfig().navbar.items;
}

function NavbarItems({items}) {
  return (
    <>
      {items.map((item, i) => (
        <ErrorCauseBoundary
          key={i}
          onError={(error) =>
            new Error(
              `A theme navbar item failed to render.
Please double-check the following navbar item (themeConfig.navbar.items) of your Docusaurus config:
${JSON.stringify(item, null, 2)}`,
              {cause: error},
            )
          }>
          <NavbarItem {...item} />
        </ErrorCauseBoundary>
      ))}
    </>
  );
}

function NavbarContentLayout({left, right}) {
  return (
    <div className="navbar__inner">
      <div className="navbar__items">{left}</div>
      <div className="navbar__items navbar__items--right">{right}</div>
    </div>
  );
}

export default function NavbarContent() {
  const mobileSidebar = useNavbarMobileSidebar();
  const items = useNavbarItems();
  // 官方 internal function，將 config 中的 items 拆分左右
  const [leftItems, rightItems] = splitNavbarItems(items);
  // 如果 config 中沒有 search item, 預設用 SearchBar
  const searchBarItem = items.find((item) => item.type === 'search');

  // === Auth 狀態 / Modal ===
  const { token, logout } = useAuth();
  const [authVisible, setAuthVisible] = useState(false);

  const userMenu = (
    <Menu
      items={[
        {
          key: 'dashboard',
          label: (
            <Link to="/dashboard" style={{ color: "inherit" }}>
              儀表板
            </Link>
          )
        },
        {
          key: 'logout',
          label: (
            <span
              onClick={() => {
                logout();
                message.success('已登出');
              }}
            >
              登出
            </span>
          ),
        },
      ]}
    />
  );

  return (
    <>
      <NavbarContentLayout
        left={
          <>
            {/* 手機版側邊欄切換按鈕 */}
            {!mobileSidebar.disabled && <NavbarMobileSidebarToggle />}
            {/* Logo */}
            <NavbarLogo />
            {/* 左側 items */}
            <NavbarItems items={leftItems} />
          </>
        }
        right={
          <>
            {/* 右側 items */}
            <NavbarItems items={rightItems} />

            {/* 顯示 Theme 切換按鈕 (若你 config 有 colorMode.enableSwitch) */}
            <NavbarColorModeToggle className={styles.colorModeToggle} />

            {/* 若 config 裡沒定義 search，預設就顯示一個 SearchBar */}
            {!searchBarItem && (
              <NavbarSearch>
                <SearchBar />
              </NavbarSearch>
            )}

            {/* 登入 / 登出 / Avatar */}
            {token ? (
              <Dropdown overlay={userMenu} placement="bottomRight">
                <Avatar
                  icon={<UserOutlined />}
                  style={{ cursor: 'pointer', backgroundColor: '#87d068' }}
                />
              </Dropdown>
            ) : (
              <Button
                shape="circle"
                icon={<UserOutlined style={{ fontSize: 18 }} />}
                onClick={() => setAuthVisible(true)}
              />
            )}
          </>
        }
      />

      {/* Auth Modal 放最外層 */}
      <AuthModal visible={authVisible} onCancel={() => setAuthVisible(false)} />
    </>
  );
}
