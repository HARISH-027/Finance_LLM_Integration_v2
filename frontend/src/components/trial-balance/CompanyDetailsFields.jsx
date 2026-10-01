import './CompanyDetailsFields.css';

const STANDARD_OPTIONS = [
  { value: '', label: 'Auto-detect' },
  { value: 'AS', label: 'AS' },
  { value: 'IND_AS', label: 'IND AS' },
];

const ERP_OPTIONS = [
  { value: '', label: 'Auto-detect' },
  { value: 'tally', label: 'Tally' },
  { value: 'sap', label: 'SAP' },
  { value: 'oracle', label: 'Oracle' },
];

/**
 * Optional Company Details block, rendered below the upload control on the
 * "Upload new" tab only (single mode's slot, and each of comparison mode's
 * CY/PY slots — never on the "Existing (Database)" tab, which already
 * resolves company/CIN/FY from the picked document).
 *
 * All four fields are independently optional: company name, CIN, and
 * financial year are plain text (no validation — "non mandatory either it
 * filled by user or not"); accounting standard is a fixed AS/IND AS choice.
 * A blank field means "keep parsing/auto-detecting as today" — see
 * ingest_tb_to_live's own docstring for the override semantics.
 *
 * `companies` (fetched once by the parent picker, shared across every slot)
 * backs the Company Name datalist; picking or typing an exact match offers
 * that company's CIN/financial years, but only into fields the user hasn't
 * already filled — it never overwrites something the user typed.
 */
export default function CompanyDetailsFields({ slot, value, onChange, companies }) {
  const v = value || {};
  const datalistId = `tbcompany-suggest-${slot}`;
  const fyDatalistId = `tbcompany-fy-${slot}`;

  const matched = v.companyName
    ? (companies || []).find((c) => c.company_name.toLowerCase() === v.companyName.trim().toLowerCase())
    : null;

  function set(field, val) {
    onChange(slot, field, val);
  }

  function onCompanyNameChange(next) {
    set('companyName', next);
    const exact = (companies || []).find((c) => c.company_name.toLowerCase() === next.trim().toLowerCase());
    if (exact) {
      if (!v.cin && exact.cin) set('cin', exact.cin);
      if (!v.financialYear && exact.financial_years?.length) set('financialYear', exact.financial_years[0]);
    }
  }

  return (
    <div className="tbcompanyfields">
      <span className="tbpick__slot-label">Company details</span>
      <div className="tbcompanyfields__grid">
        <label className="tbcompanyfields__field">
          <span>Company name</span>
          <input
            type="text"
            className="tbpick__select"
            list={datalistId}
            placeholder="e.g. Antrix Corporation Limited"
            value={v.companyName || ''}
            onChange={(e) => onCompanyNameChange(e.target.value)}
          />
          <datalist id={datalistId}>
            {(companies || []).map((c) => <option key={c.company_name} value={c.company_name} />)}
          </datalist>
        </label>

        <label className="tbcompanyfields__field">
          <span>CIN</span>
          <input
            type="text"
            className="tbpick__select"
            placeholder="e.g. U85110KA1992GOI013570"
            value={v.cin || ''}
            onChange={(e) => set('cin', e.target.value)}
          />
        </label>

        <label className="tbcompanyfields__field">
          <span>Financial year</span>
          <input
            type="text"
            className="tbpick__select"
            list={fyDatalistId}
            placeholder="e.g. 2024-25"
            value={v.financialYear || ''}
            onChange={(e) => set('financialYear', e.target.value)}
          />
          <datalist id={fyDatalistId}>
            {(matched?.financial_years || []).map((y) => <option key={y} value={y} />)}
          </datalist>
        </label>

        <label className="tbcompanyfields__field">
          <span>Accounting standard</span>
          <select
            className="tbpick__select"
            value={v.standard || ''}
            onChange={(e) => set('standard', e.target.value)}
          >
            {STANDARD_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>

        <label className="tbcompanyfields__field">
          <span>Source ERP</span>
          <select
            className="tbpick__select"
            value={v.erpType || ''}
            onChange={(e) => set('erpType', e.target.value)}
          >
            {ERP_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>
      </div>
    </div>
  );
}
