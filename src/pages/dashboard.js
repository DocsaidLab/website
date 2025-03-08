// src/pages/dashboard.js
import {
  CommentOutlined,
  DatabaseOutlined,
  HomeOutlined,
  KeyOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  PoweroffOutlined,
  UserOutlined,
} from "@ant-design/icons";
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
  message,
} from "antd";
import React, { useEffect, useMemo, useState } from "react";
import { useAuth } from "../context/AuthContext";

// Dashboard 子頁面
import DashboardApiKey from "../components/Dashboard/ApiKey";
import DashboardApiUsage from "../components/Dashboard/ApiUsage";
import DashboardMyComments from "../components/Dashboard/MyComments";
import DashboardMyInfo from "../components/Dashboard/MyInfo";

const { Header: AntHeader, Sider, Content, Footer } = AntLayout;

export default function DashboardPage() {
  const { token, user, loading, logout } = useAuth();
  const [selectedKey, setSelectedKey] = useState("myinfo");
  const [collapsed, setCollapsed] = useState(false);

  const { token: designToken } = antdTheme.useToken();

  useEffect(() => {
    if (!loading && !token) {
      message.warning("請先登入");
      // 可根據需求導向登入頁面，例如：window.location.href = "/";
    }
  }, [loading, token]);

  const contentComponent = useMemo(() => {
    switch (selectedKey) {
      case "myinfo":
        return <DashboardMyInfo />;
      case "comments":
        return <DashboardMyComments />;
      case "apikey":
        return <DashboardApiKey />;
      case "apiusage":
        return <DashboardApiUsage />;
      default:
        return null;
    }
  }, [selectedKey]);

  const pageTitle = useMemo(() => {
    switch (selectedKey) {
      case "myinfo":
        return "我的資訊";
      case "comments":
        return "我的留言";
      case "apikey":
        return "我的 API Key";
      case "apiusage":
        return "API 使用紀錄";
      default:
        return "未定義";
    }
  }, [selectedKey]);

  // 建立右上角用戶選單，使用新版 Dropdown API
  const userMenuItems = [
    {
      key: "backHome",
      icon: <HomeOutlined />,
      label: "回主站",
      onClick: () => {
        window.location.href = "/";
      },
    },
    {
      key: "logout",
      icon: <PoweroffOutlined />,
      label: "登出",
      onClick: logout,
    },
  ];

  // 使用新版 Breadcrumb API
  const breadcrumbItems = [
    { title: "我的後台" },
    { title: pageTitle },
  ];

  if (loading) {
    return (
      <Layout title="後台">
        <Spin style={{ margin: "50px auto", display: "block" }} />
      </Layout>
    );
  }

  if (!token) {
    return (
      <Layout title="後台">
        <div style={{ textAlign: "center", margin: 50 }}>
          <h2>尚未登入</h2>
        </div>
      </Layout>
    );
  }

  return (
    <Layout title="我的後台">
      <AntLayout style={{ minHeight: "calc(100vh - 60px)" }}>
        <Sider
          theme="light"
          collapsible
          collapsed={collapsed}
          onCollapse={setCollapsed}
          style={{ borderRight: "1px solid #ddd" }}
        >
          <div
            style={{
              padding: 16,
              fontWeight: "bold",
              textAlign: "center",
              borderBottom: "1px solid #eee",
            }}
          >
            {collapsed ? "後台" : "我的後台"}
          </div>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            onClick={(e) => setSelectedKey(e.key)}
            items={[
              { key: "myinfo", icon: <UserOutlined />, label: "我的資訊" },
              { key: "comments", icon: <CommentOutlined />, label: "我的留言" },
              { key: "apikey", icon: <KeyOutlined />, label: "我的 API Key" },
              { key: "apiusage", icon: <DatabaseOutlined />, label: "API 使用紀錄" },
            ]}
          />
        </Sider>
        <AntLayout>
          <AntHeader
            style={{
              background: designToken.colorBgContainer,
              borderBottom: "1px solid #ccc",
              padding: "0 16px",
            }}
          >
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
                  <div style={{ cursor: "pointer" }}>
                    Hi, {user?.username || "User"}!
                  </div>
                </Dropdown>
              </Col>
            </Row>
          </AntHeader>
          <Content style={{ padding: 16, background: "#f5f5f5" }}>
            <div
              style={{
                padding: 16,
                background: "#fff",
                boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
              }}
            >
              {contentComponent}
            </div>
          </Content>
          <Footer style={{ textAlign: "center", background: "#fff" }}>
            <div style={{ borderTop: "1px solid #eee", paddingTop: 8 }}>
              © {new Date().getFullYear()} My Company. All rights reserved.
            </div>
          </Footer>
        </AntLayout>
      </AntLayout>
    </Layout>
  );
}
