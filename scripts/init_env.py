"""Create local development settings once, without printing the generated key."""
from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / '.env'
    content = (root / '.env.example').read_text(encoding='utf-8')
    content = content.replace('API_KEY=\n', f'API_KEY={secrets.token_urlsafe(32)}\n', 1)
    try:
        with target.open('x', encoding='utf-8') as output:
            output.write(content)
    except FileExistsError:
        print('.env already exists; kept existing settings.')
    else:
        print('Created .env with a random API key. Development DEMO mode is enabled.')


if __name__ == '__main__':
    main()
