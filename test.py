import asyncio


async def test_async(i):
    while True:
        print(f"Hello from async function {i}!")
        await asyncio.sleep(1)


async def main():
    coroutines = []
    for i in range(5):
        print(f"Hello from main function! {i}")
        coroutines.append(test_async(i))

    result = await asyncio.gather(*coroutines)
    # print(result)


if __name__ == "__main__":
    asyncio.run(main())
