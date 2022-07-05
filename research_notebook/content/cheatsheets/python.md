# Python

## Credentials

```yaml
account:
  username: username
  password: password
```

```python
import yaml

file = "..."

with open(f"{file}}", "r") as file:
    credentials = yaml.full_load(file)

username = credentials["aviso"]["username"]
password = credentials["aviso"]["password"]
```