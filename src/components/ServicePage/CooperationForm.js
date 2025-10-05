import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Button, Form, Input, message, Select, Steps, Typography } from 'antd';
import { useEffect, useState } from 'react';
import './CooperationForm.css';

const { TextArea } = Input;
const { Text } = Typography;

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
      modulesCount: {
        label: { 'zh-hant': '並行專案／模組數', 'en': 'Concurrent projects/modules', 'ja': '並行プロジェクト／モジュール数' },
        placeholder: { 'zh-hant': '-- 請選擇 --', 'en': '-- Please choose --', 'ja': '-- 選択してください --' },
        hint: {
          'zh-hant': '我們可同時維護多個專案；計價為「每專案每月」。',
          'en': 'We can maintain multiple projects concurrently; pricing is per project per month.',
          'ja': '複数プロジェクトの同時保守が可能です。料金は「プロジェクト毎・月額」です。'
        }
      },
      budget: {
        label: { 'zh-hant': '預算範圍（每專案每月，TWD）', 'en': 'Budget range (per project per month, TWD)', 'ja': '予算範囲（プロジェクト毎・月額、TWD）' },
        validation: { 'zh-hant': '請選擇預算範圍', 'en': 'Please select a budget range', 'ja': '予算範囲を選択してください' },
        placeholder: { 'zh-hant': '-- 請選擇 --', 'en': '-- Please choose --', 'ja': '-- 選択してください --' },
        hint: {
          'zh-hant': '起點為 TWD 100,000／專案／月；實際報價將於需求釐清後提供（長期維運多採月保 Retainer）。',
          'en': 'Starts at TWD 100,000 per project per month; final quote after scoping (retainer is common for ongoing work).',
          'ja': '開始価格は TWD 100,000／プロジェクト／月。最終見積は要件整理後（継続案件はリテイナーが一般的）。'
        }
      },
      message: {
        label: { 'zh-hant': '需求說明 / 其他備註', 'en': 'Request details / Remarks', 'ja': '依頼内容 / その他の注意点' },
        placeholder: {
          'zh-hant': '請描述目標、資料情況、目前瓶頸與預計時程。若有多個專案，請簡述各模組與優先順序。',
          'en': 'Describe goals, data status, current bottlenecks, and timeline. If multiple projects, list modules and priorities.',
          'ja': '目標、データ状況、課題、希望タイムラインをご記入ください。複数案件の場合は各モジュールと優先度も。'
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

  // --------- Options aligned to per-project monthly retainer model ---------
  const projectTypeOptions = [
    { label: currentLocale === 'en' ? 'Not sure yet' : currentLocale === 'ja' ? '未定' : '尚不確定', value: 'Not sure' },
    {
      label: currentLocale === 'en'
        ? 'Single model module: development & long-term maintenance (per-project monthly retainer)'
        : currentLocale === 'ja'
          ? '単一モデルモジュール：開発と長期保守（月額リテイナー／プロジェクト毎）'
          : '單一模型模組：開發與長期維護（每專案月保）',
      value: 'Single model module (retainer)',
    },
    {
      label: currentLocale === 'en'
        ? 'Consulting (time-sliced, embedded collaboration)'
        : currentLocale === 'ja'
          ? 'コンサル（週タイムスライス・内製協働）'
          : '顧問合作（週工時切分・內嵌協作）',
      value: 'Consulting (time-sliced)',
    },
    {
      label: currentLocale === 'en'
        ? 'MVP: demonstrable model product from 0 (one-off scope)'
        : currentLocale === 'ja'
          ? 'MVP：ゼロからのモデル製品（単発スコープ）'
          : 'MVP 原型：從 0 打造（一次性範疇）',
      value: 'MVP from zero (one-off)',
    },
    {
      label: currentLocale === 'en'
        ? 'Other (please specify in remarks)'
        : currentLocale === 'ja'
          ? 'その他（備考に記入してください）'
          : '其他（請在備註中補充）',
      value: 'Other',
    },
  ];

  const modulesCountOptions = [
    { label: currentLocale === 'en' ? 'Not sure yet' : currentLocale === 'ja' ? '未定' : '尚不確定', value: 'Not sure' },
    { label: '1', value: '1' },
    { label: '2', value: '2' },
    { label: '3', value: '3' },
    { label: currentLocale === 'en' ? '4 or more' : currentLocale === 'ja' ? '4以上' : '4 以上', value: '4+' },
  ];

  // TWD ranges start at 100k per project per month
  const budgetOptions = [
    { label: currentLocale === 'en' ? 'Not sure yet' : currentLocale === 'ja' ? '未定' : '尚不確定', value: 'Not sure' },
    {
      label: currentLocale === 'en' ? 'TWD 100,000 – 150,000 / month / project'
        : currentLocale === 'ja' ? 'TWD 100,000 – 150,000／月／プロジェクト'
        : 'TWD 100,000 – 150,000／月／專案',
      value: 'TWD 100,000 – 150,000 / month / project',
    },
    {
      label: currentLocale === 'en' ? 'TWD 150,000 – 250,000 / month / project'
        : currentLocale === 'ja' ? 'TWD 150,000 – 250,000／月／プロジェクト'
        : 'TWD 150,000 – 250,000／月／專案',
      value: 'TWD 150,000 – 250,000 / month / project',
    },
    {
      label: currentLocale === 'en' ? 'TWD 250,000 – 400,000 / month / project'
        : currentLocale === 'ja' ? 'TWD 250,000 – 400,000／月／プロジェクト'
        : 'TWD 250,000 – 400,000／月／專案',
      value: 'TWD 250,000 – 400,000 / month / project',
    },
    {
      label: currentLocale === 'en' ? 'Above TWD 400,000 / month / project'
        : currentLocale === 'ja' ? 'TWD 400,000 以上／月／プロジェクト'
        : 'TWD 400,000 以上／月／專案',
      value: 'Above TWD 400,000 / month / project',
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
      // Keep backend payload compatible.
      // Append new fields (modulesCount + billing model) into message to avoid API breakage.
      const metaLines = [
        `modules_count: ${values.modulesCount || 'Not specified'}`,
        `billing_model: per-project monthly retainer (TWD)`,
      ];
      const mergedMessage = (values.message ? values.message + '\n\n' : '') + '---\n' + metaLines.join('\n');

      const response = await fetch('https://api.docsaid.org/cooperation/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: values.name,
          email: values.email,
          project_type: values.projectType,
          budget: values.budget,
          message: mergedMessage,
        }),
      });

      const result = await response.json();
      if (response.ok) {
        message.success(getText(translations.messages.success));
        setSubmitted(true);
        setCurrentStep(0);
      } else {
        message.error(
          getText(translations.messages.submitError) +
          (result.detail || result.message || (currentLocale === 'en' ? 'Please try again later' : currentLocale === 'ja' ? '後ほどお試しください。' : '請稍後再試'))
        );
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
          label={getText(translations.formLabels.modulesCount.label)}
          name="modulesCount"
          initialValue={'1'}
          extra={<Text type="secondary" style={{ fontSize: 12 }}>{getText(translations.formLabels.modulesCount.hint)}</Text>}
        >
          <Select placeholder={getText(translations.formLabels.modulesCount.placeholder)} options={modulesCountOptions} />
        </Form.Item>

        <Form.Item
          label={getText(translations.formLabels.budget.label)}
          name="budget"
          rules={[{ required: true, message: getText(translations.formLabels.budget.validation) }]}
          initialValue={budgetOptions[0].value}
          extra={<Text type="secondary" style={{ fontSize: 12 }}>{getText(translations.formLabels.budget.hint)}</Text>}
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
