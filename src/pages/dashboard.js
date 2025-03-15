// src/pages/dashboard.js
import {
  HomeOutlined,
  KeyOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  PoweroffOutlined,
  UserOutlined
} from "@ant-design/icons";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import Layout from "@theme/Layout";
import {
  Layout as AntLayout,
  Breadcrumb,
  Button,
  Col,
  Dropdown,
  Menu,
  Row,
  Spin,
  theme as antdTheme,
  message
} from "antd";
import React, { useEffect, useMemo, useState } from "react";
import DashboardApiKey from "../components/Dashboard/ApiKey";
import DashboardMyInfo from "../components/Dashboard/MyInfo";
import { useAuth } from "../context/AuthContext";
import styles from "./DashboardPage.module.css";

const { Header: AntHeader, Sider, Content } = AntLayout;

const localeText = {
  "zh-hant": {
    dashboardTitle: "我的後台",
    loginWarning: "請先登入",
    notLoggedIn: "尚未登入",
    sider: {
      collapsed: "後台",
      expanded: "我的後台",
      menu: {
        myinfo: "我的資訊",
        apikey: "我的 API Key"
      }
    },
    breadcrumb: {
      dashboard: "我的後台",
      myinfo: "我的資訊",
      apikey: "我的 API Key",
      undefined: "未定義"
    },
    userMenu: {
      backHome: "回主站",
      logout: "登出"
    }
  },
  en: {
    dashboardTitle: "Dashboard",
    loginWarning: "Please log in first",
    notLoggedIn: "Not logged in",
    sider: {
      collapsed: "Dashboard",
      expanded: "My Dashboard",
      menu: {
        myinfo: "My Information",
        apikey: "My API Key"
      }
    },
    breadcrumb: {
      dashboard: "Dashboard",
      myinfo: "My Information",
      apikey: "My API Key",
      undefined: "Undefined"
    },
    userMenu: {
      backHome: "Back to Site",
      logout: "Logout"
    }
  },
  ja: {
    dashboardTitle: "ダッシュボード",
    loginWarning: "まずログインしてください",
    notLoggedIn: "ログインしていません",
    sider: {
      collapsed: "ダッシュボード",
      expanded: "マイダッシュボード",
      menu: {
        myinfo: "マイ情報",
        apikey: "マイAPIキー"
      }
    },
    breadcrumb: {
      dashboard: "ダッシュボード",
      myinfo: "マイ情報",
      apikey: "マイAPIキー",
      undefined: "未定義"
    },
    userMenu: {
      backHome: "サイトへ戻る",
      logout: "ログアウト"
    }
  }
};

export default function DashboardPage() {
  const {
    i18n: { currentLocale }
  } = useDocusaurusContext();
  const text = localeText[currentLocale] || localeText.en;
  const { token, user, loading, logout } = useAuth();
  const [selectedKey, setSelectedKey] = useState("myinfo");
  const [collapsed, setCollapsed] = useState(false);
  const { token: designToken } = antdTheme.useToken();

  useEffect(() => {
    if (!loading && !token) {
      message.warning(text.loginWarning);
      // 可根據需求導向登入頁面，例如：window.location.href = "/";
    }
  }, [loading, token, text.loginWarning]);

  const contentComponent = useMemo(() => {
    switch (selectedKey) {
      case "myinfo":
        return <DashboardMyInfo />;
      case "apikey":
        return <DashboardApiKey />;
      default:
        return null;
    }
  }, [selectedKey]);

  const pageTitle = useMemo(() => {
    switch (selectedKey) {
      case "myinfo":
        return text.breadcrumb.myinfo;
      case "apikey":
        return text.breadcrumb.apikey;
      default:
        return text.breadcrumb.undefined;
    }
  }, [selectedKey, text.breadcrumb]);

  // 根據語系決定首頁路徑
  let homePath = "/";
  if (currentLocale === "en") {
    homePath = "/en";
  } else if (currentLocale === "ja") {
    homePath = "/ja";
  }

  const userMenuItems = [
    {
      key: "backHome",
      icon: <HomeOutlined />,
      label: text.userMenu.backHome,
      onClick: () => {
        window.location.href = homePath;
      }
    },
    {
      key: "logout",
      icon: <PoweroffOutlined />,
      label: text.userMenu.logout,
      onClick: logout
    }
  ];

  const breadcrumbItems = [
    { title: text.breadcrumb.dashboard },
    { title: pageTitle }
  ];

  if (loading) {
    return (
      <Layout title={text.dashboardTitle}>
        <Spin className={styles.loadingSpin} />
      </Layout>
    );
  }

  if (!token) {
    return (
      <Layout title={text.dashboardTitle}>
        <div className={styles.notLoggedInContainer}>
          <h2>{text.notLoggedIn}</h2>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title={text.dashboardTitle}>
      <AntLayout className={styles.dashboardLayout}>
        <Sider
          theme="light"
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          breakpoint="md"
          onBreakpoint={(broken) => setCollapsed(broken)}
          className={styles.customSider}
        >
          <div className={styles.siderTitle}>
            {collapsed ? text.sider.collapsed : text.sider.expanded}
          </div>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            onClick={(e) => setSelectedKey(e.key)}
            items={[
              {
                key: "myinfo",
                icon: <UserOutlined />,
                label: text.sider.menu.myinfo
              },
              {
                key: "apikey",
                icon: <KeyOutlined />,
                label: text.sider.menu.apikey
              }
            ]}
          />
        </Sider>
        <AntLayout className={styles.rightSideLayout}>
          <AntHeader className={styles.header} style={{ background: designToken.colorBgContainer }}>
            <Row align="middle" justify="space-between" style={{ height: "100%" }}>
              <Col>
                <Row align="middle" gutter={16}>
                  <Col>
                    <Button
                      type="text"
                      icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                      onClick={() => setCollapsed(!collapsed)}
                    />
                  </Col>
                  <Col>
                    <Breadcrumb items={breadcrumbItems} />
                  </Col>
                </Row>
              </Col>
              <Col>
                <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
                  <div className={styles.userMenuText}>
                    Hi, {user?.username || "User"}!
                  </div>
                </Dropdown>
              </Col>
            </Row>
          </AntHeader>
          <Content className={styles.contentArea}>
            <div className={styles.contentBox}>
              {contentComponent}
            </div>
          </Content>
        </AntLayout>
      </AntLayout>
    </Layout>
  );
}
