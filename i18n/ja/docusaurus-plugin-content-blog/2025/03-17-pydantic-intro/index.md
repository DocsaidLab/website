---
slug: pydantic-intro
title: Pydantic 入門：Python データ検証と管理
authors: Z. Yuan
image: /ja/img/2025/0317.webp
tags: [Python, Pydantic, Data Validation]
description: Pydantic の基本的な概念を簡単に紹介します。
---

最近、バックエンドを作成している時にこのツールを使用したので、使ったからには記録しておこうと思いました。

<!-- truncate -->

## Pydantic とは？

[Pydantic](https://docs.pydantic.dev) は、Python で書かれたデータ検証ツールで、さまざまな乱雑なデータの問題を解決し、ついでに隅っこでほこりをかぶっている設定ファイルも管理してくれます。このツールは特にエンジニアに適しており、以下のような問題を解決します：

- **データ検証**：変なデータをフィルタリングしてくれます。
- **型変換**：フロントエンドから送られてきた謎のデータ（例えば「"123"」）を簡単に希望する `int` 型に変換できます。
- **データモデルの定義**：プログラムをちゃんとした方法で書けるようになり、Dict を無作為に突っ込むことがなくなります。
- **API の安全性と可読性向上**：見た目が良くなります。
- **FastAPI とのシームレスな統合**：基本的には FastAPI と密接に統合されており、200 行ほどのコードを省略できます。

インストールは非常に簡単です：

```bash
pip install pydantic
```

## 基本的な使い方とよく使う機能

### 1. データモデルの作成

`BaseModel` を使って構造化データモデルを定義します：

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
```

### 2. 自動検証

`User` オブジェクトを作成すると、Pydantic は自動的にデータの検証と型変換を行います：

```python
user = User(id=1, name='Alice', email='alice@example.com')
print(user)
```

もしデータ型が一致しない場合、`ValidationError` が発生します：

```python
try:
    user = User(id='abc', name='Alice', email='alice@example.com')
except Exception as e:
    print(e)
```

### 3. 自動変換とデフォルト値

自動変換の例：

```python
user = User(id='123', name='Bob', email='bob@example.com')
print(user.id)  # 出力 123 (int)
```

デフォルト値とオプションのフィールドの例：

```python
from typing import Optional

class User(BaseModel):
    id: int
    name: str = 'Unknown'
    is_active: bool = True
    nickname: Optional[str] = None

user = User(id=10)
print(user.name)      # Unknown
print(user.is_active) # True
```

### 4. ネストされたモデル

ネストされたモデルを定義する際、Pydantic は `dict` を自動的に対応するオブジェクトに変換します：

```python
class Address(BaseModel):
    city: str
    zipcode: str

class User(BaseModel):
    id: int
    name: str
    address: Address

user = User(id=1, name="Alice", address={"city": "Taipei", "zipcode": "100"})
print(user.address.city)  # 出力 "Taipei"
```

## 高度な機能

### 1. カスタムバリデーター

`validator` を使用してフィールドの検証ロジックをカスタマイズします：

```python
from pydantic import validator

class User(BaseModel):
    id: int
    email: str

    @validator('email')
    def email_must_contain_at(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v
```

### 2. 複数フィールドの検証

`root_validator` を使って複数のフィールドを検証します：

```python
from pydantic import root_validator

class User(BaseModel):
    password1: str
    password2: str

    @root_validator
    def passwords_match(cls, values):
        if values.get('password1') != values.get('password2'):
            raise ValueError('Passwords do not match')
        return values
```

## FastAPI での利用

FastAPI と組み合わせて、Pydantic でデータモデルを定義します：

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    id: int
    name: str
    price: float

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item.dict()}
```

## 他の便利な機能

いくつかのあまり知られていないが非常に強力な使用方法があります：

### 1. ジェネリックモデル

汎用的なメッセージ（例：ステータスやエラーメッセージ）を含む API レスポンスを作成したい場合、次のように使います：

```python
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T")

class ResponseModel(BaseModel, Generic[T]):
    data: T
    message: str
```

`TypeVar` を使って型変数 `T` を定義し、将来的に置き換え可能な任意の型を表します。`ResponseModel` は `BaseModel` と `Generic[T]` を継承することで、必要に応じて異なる型のデータを受け入れることができます。この汎用モデルにより、再利用性と柔軟性が大幅に向上します。

### 2. 非同期処理

`parse_obj_as` を使って非同期データ解析を実現します。このメソッドは、リストや辞書のような元のデータ構造を指定した Pydantic モデルに変換します。大量のデータを非同期で処理する場面で役立ちます。

```python
from pydantic import parse_obj_as
from typing import List

users = parse_obj_as(List[User], [{"id": 1, "name": "Alice", "email": "alice@example.com"}])
```

`parse_obj_as` は、複数のデータを含むコレクション（この例では `List[User]`）を自動的に対応するモデルインスタンスに解析します。非同期環境でデータベースや API から取得した JSON データを処理する際に便利です。

### 3. スキーマ定義

Pydantic は自動的に JSON スキーマを生成できます。モデルの `schema_json()` メソッドを呼び出すだけです：

```python
user_schema = User.schema_json()
print(user_schema)
```

この機能は API ドキュメントの生成、データ検証、API との統合に非常に便利です。生成されたスキーマは API ドキュメントの一部として直接使用でき、フロントエンドやサードパーティの開発者が参照できます。

### 4. ORM 統合

SQLAlchemy と連携して Pydantic を使用することで、データベースモデルと API データモデル間で橋渡しを行い、データ検証と自動変換を実現します。

```python
from pydantic.orm import from_orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()

class UserORM(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String)

class UserSchema(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True
```

まず、SQLAlchemy でデータベースの ORM モデル（例えば `UserORM`）を定義し、データテーブルの構造を記述します。

対応する Pydantic モデルでは、`Config` で `orm_mode = True` を設定し、ORM オブジェクトからデータを抽出して自動的に変換・検証します。`from_orm` メソッドを使用すれば、ORM インスタンスを簡単に Pydantic モデルに変換でき、手動でフィールドをマッピングする必要がなくなります。

この統合方法は、Web API 開発に特に適しており、データベースのデータを検証した後にフロントエンドに返す際に、コードの簡略化とエラーの削減に役立ちます。

## `dataclass` との比較

Pydantic と Python の組み込み `dataclass` はどちらも構造化データモデルを定義できますが、データ検証と型変換の面で顕著な違いがあります。

以下の表に比較を示します：

<div style={{ display: "flex", justifyContent: "center" }}>

| 特徴                     | `dataclass`（組み込み）               | `pydantic.BaseModel`（外部ライブラリ）       |
| ------------------------ | ------------------------------------- | -------------------------------------------- |
| **データ検証**           | ❌ サポートなし（手動チェックが必要） | ✅ 自動検証（型とフォーマット）              |
| **型変換**               | ❌ サポートなし（手動処理が必要）     | ✅ 自動変換（例： `"123"` → `int`）          |
| **パフォーマンス**       | ⭐⭐⭐⭐（CPython ネイティブ）        | ⭐⭐⭐（v2 で Rust コア最適化）              |
| **JSON 変換**            | ❌ 手動で `json.dumps` を使用         | ✅ 内蔵の `.json()` と `.dict()`             |
| **ネストされたモデル**   | ❌ 手動でネストを処理                 | ✅ 内蔵サポート、自動解析                    |
| **オプションフィールド** | ❌ 手動で `Optional` を設定           | ✅ 内蔵サポート                              |
| **環境変数の読み取り**   | ❌ サポートなし                       | ✅ `BaseSettings` で `.env` を読み取り       |
| **適用シーン**           | 軽量データストレージ                  | API 検証、データ解析と複雑なアプリケーション |

</div>

## まとめ

Pydantic の基本知識を簡単に見てきましたが、自動検証や型変換といった機能は非常に便利ですね。

プログラムのコード量を減らすライブラリはぜひ覚えておきましょう。試してみてください。
