/*
Copyright 2026 Google LLC. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
const CONSTANT_VALUES = require('./constant');

/**
 * Invoked from csat_adkgo.yml file to post survey link
 * in closed issue.
 * @param {!Object.<string,!Object>} github contains pre defined functions.
 *  context Information about the workflow run.
 * @return {null}
 */
module.exports = async ({ github, context }) => {
  const issue = context.payload.issue.html_url;
  const baseUrl = CONSTANT_VALUES.MODULE.CSAT.BASE_URL;

  // Loop over all ths label present in issue and check if specific label is
  // present for survey link.
  for (const label of context.payload.issue.labels) {
    if (label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.BUG) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.DOCUMENTATION) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.GOOD_FIRST_ISSUE) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.ENHANCEMENT) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.QUESTION) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.GO) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.API) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.BREAKING_CHANGE) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.DEPENDENCIES) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.DUPLICATE) ||
      label.name.includes(CONSTANT_VALUES.GLOBALS.LABELS.V1_0_TRIAGE)) {

      console.log(`label-${label.name}, posting CSAT survey for issue =${issue}`);

      const yesCsat = `<a href="${baseUrl + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
        CONSTANT_VALUES.MODULE.CSAT.YES +
        CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + encodeURIComponent(issue)}"> ${CONSTANT_VALUES.MODULE.CSAT.YES}</a>`;

      const noCsat = `<a href="${baseUrl + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
        CONSTANT_VALUES.MODULE.CSAT.NO +
        CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + encodeURIComponent(issue)}"> ${CONSTANT_VALUES.MODULE.CSAT.NO}</a>`;

      const comment = CONSTANT_VALUES.MODULE.CSAT.MSG + '\n' + yesCsat + '\n' + noCsat + '\n';
      let issueNumber = context.issue.number ?? context.payload.issue.number;

      await github.rest.issues.createComment({
        issue_number: issueNumber,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: comment
      });
    }
  }
};
