import type { Handle } from "@sveltejs/kit";

import { ensureDatabaseSchema } from "$lib/server/db";

export const handle: Handle = async ({ event, resolve }) => {
  // A database reload can restore an older file while the dev server remains
  // running. Reconcile additive application columns before handling requests.
  ensureDatabaseSchema();
  return resolve(event);
};
