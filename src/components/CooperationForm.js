import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Form, Input, message, Select, Steps } from 'antd';
import React, { useEffect, useState } from 'react';
import './CooperationForm.css';

const { TextArea } = Input;

export default function CooperationForm() {
  const {
    i18n: { currentLocale },
  } = useDocusaurusContext();

  const getText = (obj) => obj[currentLocale] || obj['zh-hant'];

  const translations = {
    steps: [
      {
        title: { 'zh-hant': '送出中', 'en': 'Submitting', 'ja': '送信中' },
        description: {
          'zh-hant': '正在將您的需求送往伺服器...',
          'en': 'Sending your request to the server...',
          'ja': 'サーバーにリクエストを送信中です...'
        },
      },
      {
        title: { 'zh-hant': '分析中', 'en': 'Analyzing', 'ja': '解析中' },
        description: {
          'zh-hant': '系統正在分析您的需求...',
          'en': 'System is analyzing your request...',
          'ja': 'システムがあなたのリクエストを解析中です...'
        },
      },
      {
        title: { 'zh-hant': '完成', 'en': 'Completed', 'ja': '完了' },
        description: {
          'zh-hant': '您的需求已成功送達！',
          'en': 'Your request has been successfully submitted!',
          'ja': 'あなたのリクエストが正常に送信されました！'
        },
      },
    ],
    final: {
      heading: { 'zh-hant': '非常感謝您的填寫！', 'en': 'Thank you very much for your submission!', 'ja': 'ご記入いただき、誠にありがとうございます！' },
      text: {
        'zh-hant': '我們已收到您的需求，會在 1 ～ 2 個工作日內回覆您。',
        'en': 'We have received your request and will reply within 1 to 2 business days.',
        'ja': 'ご依頼を承りました。1～2営業日内にご連絡いたします。'
      },
      button: {
        'zh-hant': '再填一次',
        'en': 'Submit another',
        'ja': 'もう一度記入する'
      },
    },
    stepsScreen: {
      heading: { 'zh-hant': '感謝您的填寫！', 'en': 'Thank you for your submission!', 'ja': 'ご記入いただき、ありがとうございます！' },
      text: {
        'zh-hant': '系統正在處理您的需求...',
        'en': 'Processing your request...',
        'ja': 'システムがあなたのリクエストを処理中です...'
      },
    },
    formIntro: {
      text: {
        'zh-hant': '請填寫以下資訊，我會在 1 ～ 2 個工作日內回覆您。',
        'en': 'Please fill in the following information. I will reply within 1 to 2 business days.',
        'ja': '以下の情報をご記入ください。1～2営業日内にご連絡いたします。'
      },
    },
    formLabels: {
      name: {
        label: { 'zh-hant': '您的姓名/稱呼', 'en': 'Your Name', 'ja': 'あなたの名前' },
        placeholder: { 'zh-hant': '請輸入您的名稱', 'en': 'Please enter your name', 'ja': '名前を入力してください' },
        validation: { 'zh-hant': '請輸入您的姓名/稱呼', 'en': 'Please enter your name', 'ja': '名前を入力してください' },
      },
      email: {
        label: { 'zh-hant': '電子信箱', 'en': 'Email', 'ja': 'メールアドレス' },
        placeholder: { 'zh-hant': 'example@domain.com', 'en': 'example@domain.com', 'ja': 'example@domain.com' },
        validationRequired: { 'zh-hant': '請輸入您的電子信箱', 'en': 'Please enter your email', 'ja': 'メールアドレスを入力してください' },
        validationFormat: { 'zh-hant': '請輸入正確的 Email 格式', 'en': 'Please enter a valid email', 'ja': '正しいメール形式で入力してください' },
      },
      projectType: {
        label: { 'zh-hant': '想要合作的服務', 'en': 'Service to cooperate on', 'ja': 'ご依頼のサービス' },
        validation: { 'zh-hant': '請選擇服務類型', 'en': 'Please select a service type', 'ja': 'サービスの種類を選択してください' },
        placeholder: { 'zh-hant': '-- 請選擇 --', 'en': '-- Please choose --', 'ja': '-- 選択してください --' },
      },
      budget: {
        label: { 'zh-hant': '預期預算範圍', 'en': 'Expected budget range', 'ja': '予算範囲' },
        validation: { 'zh-hant': '請選擇預算範圍', 'en': 'Please select a budget range', 'ja': '予算範囲を選択してください' },
        placeholder: { 'zh-hant': '-- 請選擇 --', 'en': '-- Please choose --', 'ja': '-- 選択してください --' },
      },
      message: {
        label: { 'zh-hant': '需求說明 / 其他備註', 'en': 'Request details / Remarks', 'ja': '依頼内容 / その他の注意点' },
        placeholder: {
          'zh-hant': '請盡量描述您的需求、專案背景、預期時程等資訊。',
          'en': 'Please describe your needs, project background, and expected timeline.',
          'ja': 'ご依頼内容、プロジェクトの背景、予定のタイムラインなど、できるだけ詳しくご記入ください。'
        },
      },
      submit: {
        text: { 'zh-hant': '提交表單', 'en': 'Submit', 'ja': '送信' },
      },
    },
    messages: {
      success: {
        'zh-hant': '表單已成功提交！',
        'en': 'Form submitted successfully!',
        'ja': 'フォームが正常に送信されました！'
      },
      submitError: {
        'zh-hant': '表單提交失敗：',
        'en': 'Form submission failed: ',
        'ja': 'フォームの送信に失敗しました：'
      },
      error: {
        'zh-hant': '表單提交時發生錯誤，請稍後再試',
        'en': 'An error occurred while submitting the form. Please try again later.',
        'ja': 'フォーム送信中にエラーが発生しました。後ほどお試しください。'
      },
    },
  };

  const projectTypeOptions = [
    { label: currentLocale === 'en' ? 'Not sure' : currentLocale === 'ja' ? '未定' : '尚不確定', value: currentLocale === 'en' ? 'Not sure' : currentLocale === 'ja' ? '未定' : '尚不確定' },
    {
      label: currentLocale === 'en'
        ? 'Paper review and strategy advice'
        : currentLocale === 'ja'
          ? '論文レビューおよび戦略アドバイス'
          : '論文導讀與策略建議',
      value: currentLocale === 'en'
        ? 'Paper review and strategy advice'
        : currentLocale === 'ja'
          ? '論文レビューおよび戦略アドバイス'
          : '論文導讀與策略建議',
    },
    {
      label: currentLocale === 'en'
        ? 'AI Model API integration and deployment'
        : currentLocale === 'ja'
          ? 'AIモデルのAPI化と展開'
          : 'AI模型API化與部署',
      value: currentLocale === 'en'
        ? 'AI Model API integration and deployment'
        : currentLocale === 'ja'
          ? 'AIモデルのAPI化と展開'
          : 'AI模型API化與部署',
    },
    {
      label: currentLocale === 'en'
        ? 'Educational website development'
        : currentLocale === 'ja'
          ? '教育用ウェブサイトの構築'
          : '教學型網站建置',
      value: currentLocale === 'en'
        ? 'Educational website development'
        : currentLocale === 'ja'
          ? '教育用ウェブサイトの構築'
          : '教學型網站建置',
    },
    {
      label: currentLocale === 'en'
        ? 'AI model product MVP creation'
        : currentLocale === 'ja'
          ? 'AIモデル製品のMVP作成'
          : 'AI模型產品MVP打造',
      value: currentLocale === 'en'
        ? 'AI model product MVP creation'
        : currentLocale === 'ja'
          ? 'AIモデル製品のMVP作成'
          : 'AI模型產品MVP打造',
    },
    {
      label: currentLocale === 'en'
        ? 'Other (please specify in remarks)'
        : currentLocale === 'ja'
          ? 'その他（備考に記入してください）'
          : '其他(請在備註中說明)',
      value: currentLocale === 'en'
        ? 'Other (please specify in remarks)'
        : currentLocale === 'ja'
          ? 'その他（備考に記入してください）'
          : '其他',
    },
  ];

  const budgetOptions = [
    { label: currentLocale === 'en' ? 'Not sure' : currentLocale === 'ja' ? '未定' : '尚不確定', value: currentLocale === 'en' ? 'Not sure' : currentLocale === 'ja' ? '未定' : '尚不確定' },
    {
      label: currentLocale === 'en'
        ? 'Less than 10 cups of coffee (Below NTD 1,500)'
        : currentLocale === 'ja'
          ? '10杯未満のコーヒー（NTD 1,500以下）'
          : '10杯咖啡以下 (NTD 1,500 以下)',
      value: currentLocale === 'en'
        ? 'Less than 10 cups of coffee (Below NTD 1,500)'
        : currentLocale === 'ja'
          ? '10杯未満のコーヒー（NTD 1,500以下）'
          : '10杯咖啡以下 (NTD 1,500 以下)',
    },
    {
      label: currentLocale === 'en'
        ? '10 - 50 cups of coffee (NTD 1,500 - 8,000)'
        : currentLocale === 'ja'
          ? '10 - 50杯のコーヒー（NTD 1,500 - 8,000）'
          : '10 - 50杯咖啡 (NTD 1,500 - 8,000)',
      value: currentLocale === 'en'
        ? '10 - 50 cups of coffee (NTD 1,500 - 8,000)'
        : currentLocale === 'ja'
          ? '10 - 50杯のコーヒー（NTD 1,500 - 8,000）'
          : '10 - 50杯咖啡 (NTD 1,500 - 8,000)',
    },
    {
      label: currentLocale === 'en'
        ? '50 - 200 cups of coffee (NTD 8,000 - 30,000)'
        : currentLocale === 'ja'
          ? '50 - 200杯のコーヒー（NTD 8,000 - 30,000）'
          : '50 - 200杯咖啡 (NTD 8,000 - 30,000)',
      value: currentLocale === 'en'
        ? '50 - 200 cups of coffee (NTD 8,000 - 30,000)'
        : currentLocale === 'ja'
          ? '50 - 200杯のコーヒー（NTD 8,000 - 30,000）'
          : '50 - 200杯咖啡 (NTD 8,000 - 30,000)',
    },
    {
      label: currentLocale === 'en'
        ? '200 - 1,000 cups of coffee (NTD 30,000 - 150,000)'
        : currentLocale === 'ja'
          ? '200 - 1,000杯のコーヒー（NTD 30,000 - 150,000）'
          : '200 - 1,000杯咖啡 (NTD 30,000 - 150,000)',
      value: currentLocale === 'en'
        ? '200 - 1,000 cups of coffee (NTD 30,000 - 150,000)'
        : currentLocale === 'ja'
          ? '200 - 1,000杯のコーヒー（NTD 30,000 - 150,000）'
          : '200 - 1,000杯咖啡 (NTD 30,000 - 150,000)',
    },
    {
      label: currentLocale === 'en'
        ? 'More than 1,000 cups of coffee (Above NTD 150,000)'
        : currentLocale === 'ja'
          ? '1,000杯以上のコーヒー（NTD 150,000以上）'
          : '1,000杯咖啡以上 (NTD 150,000 以上)',
      value: currentLocale === 'en'
        ? 'More than 1,000 cups of coffee (Above NTD 150,000)'
        : currentLocale === 'ja'
          ? '1,000杯以上のコーヒー（NTD 150,000以上）'
          : '1,000杯咖啡以上 (NTD 150,000 以上)',
    },
  ];

  const [loading, setLoading] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = translations.steps.map(item => ({
    title: item.title[currentLocale],
    description: item.description[currentLocale],
  }));

  useEffect(() => {
    let timer;
    if (submitted && currentStep < steps.length) {
      timer = setTimeout(() => {
        setCurrentStep((prev) => prev + 1);
      }, 1200);
    }
    return () => clearTimeout(timer);
  }, [submitted, currentStep, steps.length]);

  const isFinishedAllSteps = submitted && currentStep >= steps.length;

  const onFinish = async (values) => {
    setLoading(true);
    try {
      const response = await fetch('https://api.docsaid.org/cooperation/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: values.name,
          email: values.email,
          project_type: values.projectType,
          budget: values.budget,
          message: values.message || '',
        }),
      });

      const result = await response.json();
      if (response.ok) {
        message.success(getText(translations.messages.success));
        setSubmitted(true);
        setCurrentStep(0);
      } else {
        message.error(getText(translations.messages.submitError) + (result.detail || result.message || (currentLocale === 'en' ? 'Please try again later' : currentLocale === 'ja' ? '後ほどお試しください。' : '請稍後再試')));
      }
    } catch (error) {
      console.error(error);
      message.error(getText(translations.messages.error));
    } finally {
      setLoading(false);
    }
  };

  const onFinishFailed = (errorInfo) => {
    console.log('Form failed:', errorInfo);
  };

  if (isFinishedAllSteps) {
    return (
      <div className="cooperation-form-wrapper">
        <h2>{getText(translations.final.heading)}</h2>
        <p>{getText(translations.final.text)}</p>
        <Button
          type="primary"
          onClick={() => {
            setSubmitted(false);
            setCurrentStep(0);
          }}
        >
          {getText(translations.final.button)}
        </Button>
      </div>
    );
  }

  if (submitted && !isFinishedAllSteps) {
    return (
      <div className="cooperation-steps-wrapper">
        <h3>{getText(translations.stepsScreen.heading)}</h3>
        <p>{getText(translations.stepsScreen.text)}</p>
        <Steps
          direction="vertical"
          current={currentStep}
          items={steps.map((s) => ({
            title: s.title,
            description: s.description,
          }))}
        />
      </div>
    );
  }

  return (
    <div className="cooperation-form-wrapper">
      <p>{getText(translations.formIntro.text)}</p>
      <Form
        name="cooperationForm"
        layout="vertical"
        onFinish={onFinish}
        onFinishFailed={onFinishFailed}
        className="cooperation-form"
      >
        <Form.Item
          label={getText(translations.formLabels.name.label)}
          name="name"
          rules={[{ required: true, message: getText(translations.formLabels.name.validation) }]}
        >
          <Input placeholder={getText(translations.formLabels.name.placeholder)} />
        </Form.Item>

        <Form.Item
          label={getText(translations.formLabels.email.label)}
          name="email"
          rules={[
            { required: true, message: getText(translations.formLabels.email.validationRequired) },
            { type: 'email', message: getText(translations.formLabels.email.validationFormat) },
          ]}
        >
          <Input placeholder={getText(translations.formLabels.email.placeholder)} autoComplete="email" />
        </Form.Item>

        <Form.Item
          label={getText(translations.formLabels.projectType.label)}
          name="projectType"
          rules={[{ required: true, message: getText(translations.formLabels.projectType.validation) }]}
          initialValue={projectTypeOptions[0].value}
        >
          <Select placeholder={getText(translations.formLabels.projectType.placeholder)} options={projectTypeOptions} />
        </Form.Item>

        <Form.Item
          label={getText(translations.formLabels.budget.label)}
          name="budget"
          rules={[{ required: true, message: getText(translations.formLabels.budget.validation) }]}
          initialValue={budgetOptions[0].value}
        >
          <Select placeholder={getText(translations.formLabels.budget.placeholder)} options={budgetOptions} />
        </Form.Item>

        <Form.Item label={getText(translations.formLabels.message.label)} name="message">
          <TextArea rows={5} placeholder={getText(translations.formLabels.message.placeholder)} />
        </Form.Item>

        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            loading={loading}
            disabled={loading}
          >
            {getText(translations.formLabels.submit.text)}
          </Button>
        </Form.Item>
      </Form>
    </div>
  );
}
