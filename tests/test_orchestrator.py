from agentbot.contracts.commands import Command, CommandStatus


async def test_orchestrator_runs_plan_sequentially_with_retry(fake_env):
    fake_env.records.add_command(Command(command_id="cmd_t", text="do A then B"))
    fake_env.command_queue.put(Command(command_id="cmd_t", text="do A then B"))
    await fake_env.orchestrator.run_one()
    cmd = fake_env.records.command_detail("cmd_t")
    assert cmd["command"]["status"] == CommandStatus.DONE.value
    runs = cmd["skill_runs"]
    assert [r["skill"] for r in runs] == ["A", "A", "B"]      # A retried once then B
    assert runs[0]["success"] == 0 and runs[1]["success"] == 1 and runs[2]["success"] == 1
