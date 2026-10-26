import asyncio
import os
import warnings

import aiohttp
import async_timeout

from loguru import logger
from urllib3.exceptions import InsecureRequestWarning

headers = {
    'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 "
                  "Safari/537.36",
    'accept-language': "zh-CN,zh;q=0.9,en-CN;q=0.8,en;q=0.7"
}

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None


async def request_one_url(session, url):
    try:
        async with async_timeout.timeout(15):
            response = await session.get(url, headers=headers, proxy=proxy)
            html = await response.text(errors="ignore")
            return html
    except Exception:
        pass

    cmd = f"curl --insecure --connect-timeout 10 {url}"
    proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        stdout, stderr = "", "timeout"
    
    if proc.returncode != 0:
        logger.warning(f"curl {url} return {proc.returncode}")
    
    if stdout:
        html = stdout.decode(encoding="UTF-8", errors="ignore")
    else:
        html = ""

    return html

html_texts = []
async def main(urls):
    global html_texts
    async with aiohttp.ClientSession() as session:
        tasks = [request_one_url(session, url) for url in urls]
        html_texts = await asyncio.gather(*tasks)

def fetch(urls):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(urls))
    return {urls[i]: html_texts[i] for i in range(len(urls))}


