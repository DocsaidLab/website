import { UserOutlined } from '@ant-design/icons';
import Link from "@docusaurus/Link";
import { Avatar, Button, Dropdown, message } from 'antd';
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

import AuthModal from '../../../components/AuthModal';

function useNavbarItems() {
  return useThemeConfig().navbar.items;
}

function NavbarItems({items}) {
  return (
    <>
      {items.map((item, i) => (
        <ErrorCauseBoundary key={i}>
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
  const [leftItems, rightItems] = splitNavbarItems(items);
  const searchBarItem = items.find((item) => item.type === 'search');

  const { token, user, logout } = useAuth();
  const [authVisible, setAuthVisible] = useState(false);

  const userMenuItems = [
    {
      key: 'dashboard',
      label: (
        <Link to="/dashboard" style={{ color: "inherit" }}>
          儀表板
        </Link>
      ),
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
  ];

  return (
    <>
      <NavbarContentLayout
        left={
          <>
            {!mobileSidebar.disabled && <NavbarMobileSidebarToggle />}
            <NavbarLogo />
            <NavbarItems items={leftItems} />
          </>
        }
        right={
          <>
            <NavbarItems items={rightItems} />
            <NavbarColorModeToggle className={styles.colorModeToggle} />
            {!searchBarItem && (
              <NavbarSearch>
                <SearchBar />
              </NavbarSearch>
            )}
            {token ? (
              <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
                <Avatar
                  src={user?.avatar}
                  icon={!user?.avatar ? <UserOutlined /> : undefined}
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

      <AuthModal visible={authVisible} onCancel={() => setAuthVisible(false)} />
    </>
  );
}
