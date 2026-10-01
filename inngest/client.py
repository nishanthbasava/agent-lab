import logging
import inngest

inngest_client = inngest.Inngest(
    app_id="grader",
    logger=logging.getLogger("uvicorn"),
)

@inngest_client.create_function(
    fn_id="grade-submission",
    trigger=inngest.TriggerEvent(event="submission/created"),
    retries=3,
    concurrency=[inngest.Concurrency(limit=5)],
)
async def grade_submission(ctx: inngest.Context) -> dict:
    data = ctx.event.data  # the payload you sent

    async def run_grader():
        return frizzle_submission_grader_workflow(data)

    result = await ctx.step.run("run-grader", run_grader)
    return result