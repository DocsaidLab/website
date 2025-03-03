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
import Layout from "@theme/Layout"; // Docusaurus Layout
import {
  Layout as AntLayout,
  Avatar,
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
import React, { useEffect, useState } from "react";

import { useAuth } from "../context/AuthContext";

// Dashboard 子頁面
import DashboardApiKey from "../components/Dashboard/ApiKey";
import DashboardApiUsage from "../components/Dashboard/ApiUsage";
import DashboardMyComments from "../components/Dashboard/MyComments";
import DashboardMyInfo from "../components/Dashboard/MyInfo";

const { Header: AntHeader, Sider, Content, Footer } = AntLayout;
const { useToken } = antdTheme; // antd v5 提供的 useToken Hook，可讀取設計 tokens

export default function DashboardPage() {
  const { token, user, loading, logout } = useAuth();
  const [fetching, setFetching] = useState(false);

  // 預設選單指向 "myinfo"
  const [selectedKey, setSelectedKey] = useState("myinfo");

  // 側邊欄折疊狀態
  const [collapsed, setCollapsed] = useState(false);

  const { token: designToken } = useToken();
  // 這裡可讀取 antd 的設計 Token，例如 designToken.colorBgContainer

  useEffect(() => {
    if (!loading && !token) {
      message.warning("請先登入");
      // 例如：window.location.href = "/";
    }
  }, [loading, token]);

  const renderContent = () => {
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
  };

  // 顯示頂端「麵包屑 / 頁面標題」示例（若需要更動態可再擴充）
  const pageTitle = (() => {
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
  })();

  // 建立右上角用戶選單
  const userMenu = (
    <Menu
      items={[
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
          onClick: () => {
            logout();
          },
        },
      ]}
    />
  );

  if (loading || fetching) {
    return (
      <Layout title="後台">
        <Spin
          tip="載入中..."
          style={{ margin: "50px auto", display: "block" }}
        />
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
        {/* 可伸縮的側邊欄 */}
        <Sider
          theme="light"
          collapsible
          collapsed={collapsed}
          onCollapse={(value) => setCollapsed(value)}
          style={{
            borderRight: "1px solid #ddd",
          }}
        >
          {/* Logo 或標題區塊 */}
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
              {
                key: "myinfo",
                icon: <UserOutlined />,
                label: "我的資訊",
              },
              {
                key: "comments",
                icon: <CommentOutlined />,
                label: "我的留言",
              },
              {
                key: "apikey",
                icon: <KeyOutlined />,
                label: "我的 API Key",
              },
              {
                key: "apiusage",
                icon: <DatabaseOutlined />,
                label: "API 使用紀錄",
              },
            ]}
          />
        </Sider>

        <AntLayout>
          {/* 頂欄 */}
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
                  {/* 折疊按鈕 */}
                  <Col>
                    <Button
                      type="text"
                      icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                      onClick={() => setCollapsed(!collapsed)}
                    />
                  </Col>
                  {/* 簡易麵包屑 + 頁面標題 */}
                  <Col>
                    <Breadcrumb style={{ margin: "8px 0" }}>
                      <Breadcrumb.Item>我的後台</Breadcrumb.Item>
                      <Breadcrumb.Item>{pageTitle}</Breadcrumb.Item>
                    </Breadcrumb>
                  </Col>
                </Row>
              </Col>

              <Col>
                <Dropdown overlay={userMenu} placement="bottomRight">
                  <div style={{ cursor: "pointer" }}>
                    <Avatar
                      style={{ backgroundColor: "#87d068", marginRight: 8 }}
                      // src={user?.avatarUrl} // 可放用戶頭像
                    >
                      {user?.username?.[0]?.toUpperCase() || "U"}
                    </Avatar>
                    Hi, {user?.username || "User"}!
                  </div>
                </Dropdown>
              </Col>
            </Row>
          </AntHeader>

          {/* 主要內容區塊 */}
          <Content style={{ padding: 16, background: "#f5f5f5" }}>
            <div
              style={{
                padding: 16,
                background: "#fff",
                boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
              }}
            >
              {renderContent()}
            </div>
          </Content>

          {/* Footer */}
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
