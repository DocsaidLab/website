---
slug: pydantic-intro
title: Pydantic 入門：Python 資料驗證與管理
authors: Z. Yuan
image: /img/2025/0317.webp
tags: [Python, Pydantic, Data Validation]
description: 簡單介紹 Pydantic 的基本概念。
---

前陣子寫後台的時候用到這個工具，既然用了就得記錄一下。

<!-- truncate -->

## 什麼是 Pydantic？

[Pydantic](https://docs.pydantic.dev) 就是一個用 Python 寫的驗證工具，專治各種亂七八糟的資料問題，順便幫你管管那些在角落生灰塵的設定檔案。這東西特別適合工程師，因為它能解決許多問題：

- **驗證資料**：幫你篩掉各種莫名其妙的錯誤資料。
- **型別轉換**：能把前端給你的鬼資料（例如「"123"」）輕鬆轉成你要的 `int`。
- **定義資料模型**：你可以用正經方式寫程式，不再隨便亂塞 Dict。
- **提升 API 安全性與可讀性**：就是好看。
- **和 FastAPI 無痛整合**：基本上就是跟 FastAPI 綁死了，少寫兩百行程式碼。

安裝方式很簡單，如下：

```bash
pip install pydantic
```

## 基本使用與常見功能

### 1. 建立資料模型

利用 `BaseModel` 定義結構化資料模型：

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
```

### 2. 自動驗證

建立 `User` 物件時，Pydantic 自動進行資料驗證與型別轉換：

```python
user = User(id=1, name='Alice', email='alice@example.com')
print(user)
```

若資料類型不符，會拋出 `ValidationError`：

```python
try:
    user = User(id='abc', name='Alice', email='alice@example.com')
except Exception as e:
    print(e)
```

### 3. 自動轉型與預設值

自動轉換範例：

```python
user = User(id='123', name='Bob', email='bob@example.com')
print(user.id)  # 輸出 123 (int)
```

設定預設值與選用欄位示例：

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

### 4. 巢狀模型

定義巢狀模型時，Pydantic 可自動解析 `dict` 為對應物件：

```python
class Address(BaseModel):
    city: str
    zipcode: str

class User(BaseModel):
    id: int
    name: str
    address: Address

user = User(id=1, name="Alice", address={"city": "Taipei", "zipcode": "100"})
print(user.address.city)  # 輸出 "Taipei"
```

## 進階功能

### 1. 自訂驗證器

利用 `validator` 自定義欄位驗證邏輯：

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

### 2. 跨欄位驗證

使用 `root_validator` 進行跨欄位驗證：

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

## 在 FastAPI 的應用

結合 FastAPI 使用 Pydantic 定義資料模型：

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

## 還有更多

還有一些比較少見但很厲害的使用方式：

### 1. 泛型模型

當你希望一個 API 回應既包含通用的訊息（如狀態或錯誤訊息），可以這樣用：

```python
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T")

class ResponseModel(BaseModel, Generic[T]):
    data: T
    message: str
```

使用 `TypeVar` 定義一個型別變數 `T`，表示未來可以替換的任意型別。讓 `ResponseModel` 同時繼承 `BaseModel` 與 `Generic[T]`，使其能根據需求接受不同型別的 data。這種泛型模型能夠顯著提升重用性與彈性

### 2. 異步處理

透過 `parse_obj_as` 實現異步資料解析，這個方法可以將一個原始的資料結構（如列表或字典）轉換成指定的 Pydantic 模型，適用於大量資料的非同步處理情境。

```python
from pydantic import parse_obj_as
from typing import List

users = parse_obj_as(List[User], [{"id": 1, "name": "Alice", "email": "alice@example.com"}])
```

`parse_obj_as` 能夠將包含多筆資料的集合（此例為 `List[User]`）自動解析成相對應的模型實例，避免手動逐一建立物件。

雖然此方法本身並非 async 函式，但在異步環境中處理從資料庫或 API 獲取的 JSON 資料時，可以快速轉換並進行後續驗證。利用此功能可確保傳入資料符合定義的結構與型別，有助於提升程式健壯性與資料處理效率。

### 3. Schema 定義

Pydantic 能夠自動生成 JSON Schema，直接呼叫模型的 `schema_json()` 方法即可：

```python
user_schema = User.schema_json()
print(user_schema)
```

這項功能在 API 文件生成、資料驗證以及與 API 的整合上非常實用。生成的 schema 可直接作為 API 文檔的一部分，方便前端或第三方開發者參考。

### 4. ORM 整合

結合 SQLAlchemy 使用 Pydantic，可以在資料庫模型與 API 資料模型之間建立橋樑，實現資料驗證與自動轉換，極大地簡化資料處理流程。

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

首先，透過 SQLAlchemy 定義資料庫中的 ORM 模型（如 UserORM），描述資料表結構。

在對應的 Pydantic 模型中，透過 Config 設定 `orm_mode = True`，使模型能夠從 ORM 物件中提取資料，進行自動轉換與驗證。使用 `from_orm` 方法，能夠輕鬆將 ORM 實例轉換成 Pydantic 模型，避免手動映射欄位，提升資料處理的安全性與效率。

這種整合方式特別適用於 Web API 開發，當需要將資料庫中的資料經過驗證後返回給前端時，能夠大幅簡化程式碼並減少錯誤。

## 與 `dataclass` 的比較

Pydantic 與 Python 內建的 `dataclass` 都能定義結構化數據模型，但在資料驗證與型別轉換上有明顯差異。

各項比較差異如下表：

<div style={{ display: "flex", justifyContent: "center" }}>

| 特性             | `dataclass`（內建）        | `pydantic.BaseModel`（外部庫）      |
| ---------------- | -------------------------- | ----------------------------------- |
| **資料驗證**     | ❌ 不支援（需手動檢查）    | ✅ 自動驗證（型別與格式）           |
| **型別轉換**     | ❌ 不支援（需手動處理）    | ✅ 自動轉換（例如 `"123"` → `int`） |
| **效能**         | ⭐⭐⭐⭐（CPython 原生）   | ⭐⭐⭐（v2 引入 Rust 核心優化）     |
| **JSON 轉換**    | ❌ 需手動使用 `json.dumps` | ✅ 內建 `.json()` 與 `.dict()`      |
| **巢狀模型**     | ❌ 需手動嵌套處理          | ✅ 內建支援，自動解析               |
| **選用欄位**     | ❌ 需手動設定 `Optional`   | ✅ 內建支援                         |
| **環境變數讀取** | ❌ 不支援                  | ✅ 支援 `BaseSettings` 讀取 `.env`  |
| **適用場景**     | 輕量級數據儲存             | API 驗證、數據解析與複雜應用        |

</div>

## 小結

簡單看完 Pydantic 的基本知識，是不是也覺得自動化的驗證、型別轉換這些功能很不錯？

可以讓我們少寫程式碼的套件肯定要學起來的，一起來用用看吧。
