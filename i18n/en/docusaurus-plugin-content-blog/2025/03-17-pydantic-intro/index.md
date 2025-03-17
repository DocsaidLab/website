---
slug: pydantic-intro
title: "Pydantic Introduction: Python Data Validation and Management"
authors: Z. Yuan
image: /en/img/2025/0317.webp
tags: [Python, Pydantic, Data Validation]
description: A simple introduction to the basic concepts of Pydantic.
---

I recently used this tool while working on the backend, so I thought I should document it.

<!-- truncate -->

## What is Pydantic?

[Pydantic](https://docs.pydantic.dev) is a validation tool written in Python, designed to handle various messy data issues and also help manage configuration files that often gather dust in the corner. It's particularly suitable for engineers because it solves many problems:

- **Data Validation**: Helps you filter out various incorrect data.
- **Type Conversion**: Easily converts the weird data from the front-end (e.g., `"123"`) into the desired `int`.
- **Defining Data Models**: You can write structured code without randomly inserting Dicts.
- **Enhancing API Security and Readability**: It just looks good.
- **Seamless Integration with FastAPI**: It's tightly integrated with FastAPI, saving you from writing hundreds of lines of code.

Installation is straightforward:

```bash
pip install pydantic
```

## Basic Usage and Common Features

### 1. Creating Data Models

Define a structured data model using `BaseModel`:

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
```

### 2. Automatic Validation

When creating a `User` object, Pydantic automatically validates the data and converts types:

```python
user = User(id=1, name='Alice', email='alice@example.com')
print(user)
```

If the data type does not match, it raises a `ValidationError`:

```python
try:
    user = User(id='abc', name='Alice', email='alice@example.com')
except Exception as e:
    print(e)
```

### 3. Automatic Type Conversion and Default Values

Example of automatic type conversion:

```python
user = User(id='123', name='Bob', email='bob@example.com')
print(user.id)  # Outputs 123 (int)
```

Example of default values and optional fields:

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

### 4. Nested Models

When defining nested models, Pydantic can automatically parse a `dict` into the corresponding object:

```python
class Address(BaseModel):
    city: str
    zipcode: str

class User(BaseModel):
    id: int
    name: str
    address: Address

user = User(id=1, name="Alice", address={"city": "Taipei", "zipcode": "100"})
print(user.address.city)  # Outputs "Taipei"
```

## Advanced Features

### 1. Custom Validators

Use `validator` to define custom validation logic for fields:

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

### 2. Cross-field Validation

Use `root_validator` to perform cross-field validation:

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

## Using Pydantic with FastAPI

Combine FastAPI with Pydantic to define data models:

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

## And More

Here are some less common but powerful uses:

### 1. Generic Models

When you want an API response that includes both generic messages (like status or error messages), you can use it this way:

```python
from typing import TypeVar, Generic
from pydantic import BaseModel

T = TypeVar("T")

class ResponseModel(BaseModel, Generic[T]):
    data: T
    message: str
```

By using `TypeVar`, you define a type variable `T` that can be replaced with any type. This allows `ResponseModel` to inherit from both `BaseModel` and `Generic[T]`, making it accept different types of `data` as needed. This kind of generic model greatly enhances reusability and flexibility.

### 2. Asynchronous Processing

Use `parse_obj_as` for asynchronous data parsing, which allows you to convert raw data structures (like lists or dictionaries) into specified Pydantic models, suitable for handling large amounts of data asynchronously.

```python
from pydantic import parse_obj_as
from typing import List

users = parse_obj_as(List[User], [{"id": 1, "name": "Alice", "email": "alice@example.com"}])
```

Although `parse_obj_as` itself is not an async function, it can be used in asynchronous environments to quickly parse and validate data fetched from databases or APIs. This ensures that the incoming data conforms to the defined structure and type, improving program robustness and data processing efficiency.

### 3. Schema Definition

Pydantic can automatically generate a JSON schema by calling the model's `schema_json()` method:

```python
user_schema = User.schema_json()
print(user_schema)
```

This feature is very useful for generating API documentation, data validation, and integration with APIs. The generated schema can be included as part of API documentation, making it easier for front-end developers or third-party developers to reference.

### 4. ORM Integration

Integrating Pydantic with SQLAlchemy allows you to bridge the gap between database models and API data models, enabling data validation and automatic conversion, simplifying the data handling process.

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

First, define the ORM model (like `UserORM`) in SQLAlchemy, describing the table structure. In the corresponding Pydantic model, set `orm_mode = True` in the `Config` class to allow the model to extract data from ORM objects and perform automatic conversion and validation. The `from_orm` method makes it easy to convert ORM instances into Pydantic models, avoiding manual field mappings and improving data handling security and efficiency.

This integration method is particularly useful in web API development, where you need to return validated data from a database to the front-end, significantly simplifying the code and reducing errors.

## Comparison with `dataclass`

Both Pydantic and Python's built-in `dataclass` allow you to define structured data models, but there are clear differences in data validation and type conversion.

Here’s a comparison table:

<div style={{ display: "flex", justifyContent: "center" }}>

| Feature                          | `dataclass` (built-in)                 | `pydantic.BaseModel` (external library)                |
| -------------------------------- | -------------------------------------- | ------------------------------------------------------ |
| **Data Validation**              | ❌ Not supported (manual check needed) | ✅ Automatic validation (types and formats)            |
| **Type Conversion**              | ❌ Not supported (manual handling)     | ✅ Automatic conversion (e.g., `"123"` → `int`)        |
| **Performance**                  | ⭐⭐⭐⭐ (CPython native)              | ⭐⭐⭐ (v2 introduces Rust core optimizations)         |
| **JSON Conversion**              | ❌ Requires manual `json.dumps`        | ✅ Built-in `.json()` and `.dict()`                    |
| **Nested Models**                | ❌ Requires manual nesting             | ✅ Built-in support, automatic parsing                 |
| **Optional Fields**              | ❌ Requires manual `Optional`          | ✅ Built-in support                                    |
| **Environment Variable Reading** | ❌ Not supported                       | ✅ Supports `BaseSettings` for `.env` reading          |
| **Use Case**                     | Lightweight data storage               | API validation, data parsing, and complex applications |

</div>

## Conclusion

After going through the basics of Pydantic, doesn't the automatic validation and type conversion seem really useful?

A package that helps us write less code is definitely worth learning. Let’s give it a try together!
