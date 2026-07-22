import { createHash } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

import { eq } from "drizzle-orm";

import { db } from "$lib/server/db";
import { nnTrainingRuns } from "$lib/server/db/schema";

import { clearArtifactPathsForRuns, getRunsForArtifactCleanup } from "./runs";
import type { EnsembleWeights } from "./ensemble";
import type { ReliabilityGateDiagnostics } from "./backtest";

const ARTIFACT_DIR = resolve(process.cwd(), "artifacts", "nn-runs");

type SerializableWeight = {
  name: string;
  shape: number[];
  values: number[];
};

export type SerializedModelArtifact = {
  runId: number;
  createdAt: string;
  modelFamily: string;
  hiddenLayers: number[];
  inputSize: number;
  outputSize: number;
  outputNumbers: number[];
  weights: SerializableWeight[];
  ensembleWeights?: EnsembleWeights;
  ensembleVersion?: string;
  ensembleReliability?: ReliabilityGateDiagnostics;
  inputEncoding?: "sorted_scalar_v1" | "windowed_multi_hot_v1";
  lossVersion?: string;
  trainingSeed?: number;
};

function checksumSha256(content: string): string {
  return createHash("sha256").update(content, "utf8").digest("hex");
}

export async function saveModelArtifact(
  artifact: SerializedModelArtifact,
): Promise<{
  artifactPath: string;
  sha256: string;
}> {
  await mkdir(ARTIFACT_DIR, { recursive: true });

  const artifactPath = resolve(ARTIFACT_DIR, `run-${artifact.runId}.json`);
  const content = JSON.stringify(artifact);
  const sha256 = checksumSha256(content);

  await writeFile(artifactPath, content, "utf8");

  db.update(nnTrainingRuns)
    .set({
      modelArtifactPath: artifactPath,
      modelSha256: sha256,
    })
    .where(eq(nnTrainingRuns.id, artifact.runId))
    .run();

  return { artifactPath, sha256 };
}

export async function verifyArtifactChecksum(
  artifactPath: string,
  expectedSha256: string,
): Promise<boolean> {
  const raw = await readFile(artifactPath, "utf8");
  const actual = checksumSha256(raw);
  return actual === expectedSha256;
}

export async function cleanupOldArtifacts(keepCount = 20): Promise<void> {
  const candidates = getRunsForArtifactCleanup(keepCount);
  if (candidates.length === 0) {
    return;
  }

  const removedRunIds: number[] = [];

  for (const row of candidates) {
    if (!row.modelArtifactPath) {
      continue;
    }

    const pathToRemove = row.modelArtifactPath;
    try {
      await rm(pathToRemove, { force: true });
      removedRunIds.push(row.id);
    } catch {
      // Keep DB pointers intact if file removal fails.
    }
  }

  clearArtifactPathsForRuns(removedRunIds);
}
