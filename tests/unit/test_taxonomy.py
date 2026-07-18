"""Tests for taxotreeset.taxonomy — TaxID resolution via taxoniq and NCBI CLI."""

import json
from unittest.mock import MagicMock, patch

import pytest
from taxotreeset.taxonomy import _resolve_name_via_ncbi, resolve_to_taxid


# ---------------------------------------------------------------------------
# resolve_to_taxid — numeric passthrough
# ---------------------------------------------------------------------------


class TestResolveToTaxidNumeric:
    def test_pure_digits_returned_unchanged(self):
        assert resolve_to_taxid("10239") == "10239"

    def test_leading_trailing_whitespace_stripped_before_check(self):
        assert resolve_to_taxid("  2697049  ") == "2697049"

    def test_single_digit_is_valid(self):
        assert resolve_to_taxid("1") == "1"


# ---------------------------------------------------------------------------
# resolve_to_taxid — taxoniq path
# ---------------------------------------------------------------------------


class TestResolveToTaxidViaTaxoniq:
    def test_known_name_resolved_via_taxoniq(self):
        mock_taxon = MagicMock()
        mock_taxon.tax_id = 10239
        with patch("taxotreeset.taxonomy.taxoniq.Taxon", return_value=mock_taxon):
            result = resolve_to_taxid("Viruses")
        assert result == "10239"

    def test_tax_id_cast_to_string(self):
        mock_taxon = MagicMock()
        mock_taxon.tax_id = 11118
        with patch("taxotreeset.taxonomy.taxoniq.Taxon", return_value=mock_taxon):
            result = resolve_to_taxid("Coronaviridae")
        assert result == "11118"
        assert isinstance(result, str)

    def test_keyerror_falls_through_to_ncbi(self):
        with (
            patch("taxotreeset.taxonomy.taxoniq.Taxon", side_effect=KeyError),
            patch(
                "taxotreeset.taxonomy._resolve_name_via_ncbi", return_value="10239"
            ),
        ):
            result = resolve_to_taxid("Viruses")
        assert result == "10239"

    def test_taxoniq_exception_falls_through_to_ncbi(self):
        import taxoniq

        with (
            patch(
                "taxotreeset.taxonomy.taxoniq.Taxon",
                side_effect=taxoniq.TaxoniqException,
            ),
            patch(
                "taxotreeset.taxonomy._resolve_name_via_ncbi", return_value="10239"
            ),
        ):
            result = resolve_to_taxid("Viruses")
        assert result == "10239"


# ---------------------------------------------------------------------------
# resolve_to_taxid — error path
# ---------------------------------------------------------------------------


class TestResolveToTaxidErrors:
    def test_raises_valueerror_when_both_sources_fail(self):
        with (
            patch("taxotreeset.taxonomy.taxoniq.Taxon", side_effect=KeyError),
            patch("taxotreeset.taxonomy._resolve_name_via_ncbi", return_value=None),
        ):
            with pytest.raises(ValueError, match="Could not resolve"):
                resolve_to_taxid("NotARealCladeName")

    def test_error_message_includes_reference_name(self):
        with (
            patch("taxotreeset.taxonomy.taxoniq.Taxon", side_effect=KeyError),
            patch("taxotreeset.taxonomy._resolve_name_via_ncbi", return_value=None),
        ):
            with pytest.raises(ValueError, match="FakeTaxon"):
                resolve_to_taxid("FakeTaxon")


# ---------------------------------------------------------------------------
# _resolve_name_via_ncbi
# ---------------------------------------------------------------------------


class TestResolveNameViaNcbi:
    def _mock_run(self, json_payload: dict):
        mock_result = MagicMock()
        mock_result.stdout = json.dumps(json_payload) + "\n"
        return mock_result

    def test_returns_taxid_from_json_output(self):
        payload = {"taxonomy": {"tax_id": 10239}}
        with patch(
            "taxotreeset.taxonomy.subprocess.run",
            return_value=self._mock_run(payload),
        ):
            result = _resolve_name_via_ncbi("Viruses")
        assert result == "10239"

    def test_taxid_cast_to_string(self):
        payload = {"taxonomy": {"tax_id": 11118}}
        with patch(
            "taxotreeset.taxonomy.subprocess.run",
            return_value=self._mock_run(payload),
        ):
            result = _resolve_name_via_ncbi("Coronaviridae")
        assert isinstance(result, str)

    def test_returns_none_when_stdout_is_empty(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("UnknownClade")
        assert result is None

    def test_returns_none_when_json_lacks_taxonomy_key(self):
        mock_result = MagicMock()
        mock_result.stdout = '{"other_key": {}}\n'
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("UnknownClade")
        assert result is None

    def test_returns_none_when_json_lacks_tax_id(self):
        mock_result = MagicMock()
        mock_result.stdout = '{"taxonomy": {"name": "Foo"}}\n'
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("Foo")
        assert result is None

    def test_returns_none_when_taxonomy_is_null(self):
        # "taxonomy": null (present but null) must be skipped, not crash on
        # None.get(...).
        mock_result = MagicMock()
        mock_result.stdout = '{"taxonomy": null}\n'
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("Foo")
        assert result is None

    def test_returns_none_when_line_is_not_an_object(self):
        # A non-object JSON line (bare string/array/number) must be skipped.
        mock_result = MagicMock()
        mock_result.stdout = '"just a string"\n[1, 2, 3]\n42\n'
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("Foo")
        assert result is None

    def test_returns_none_on_subprocess_calledprocesserror(self):
        import subprocess

        with patch(
            "taxotreeset.taxonomy.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "datasets"),
        ):
            result = _resolve_name_via_ncbi("Viruses")
        assert result is None

    def test_returns_none_on_oserror(self):
        with patch(
            "taxotreeset.taxonomy.subprocess.run", side_effect=OSError("no such file")
        ):
            result = _resolve_name_via_ncbi("Viruses")
        assert result is None

    def test_skips_blank_lines_in_output(self):
        mock_result = MagicMock()
        mock_result.stdout = "\n\n" + json.dumps({"taxonomy": {"tax_id": 10239}}) + "\n\n"
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("Viruses")
        assert result == "10239"

    def test_skips_non_json_lines_and_continues(self):
        mock_result = MagicMock()
        mock_result.stdout = (
            "Downloading...\n"
            + json.dumps({"taxonomy": {"tax_id": 10239}})
            + "\n"
        )
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("Viruses")
        assert result == "10239"

    def test_returns_first_taxid_from_multi_line_output(self):
        mock_result = MagicMock()
        mock_result.stdout = (
            json.dumps({"taxonomy": {"tax_id": 10239}})
            + "\n"
            + json.dumps({"taxonomy": {"tax_id": 11118}})
            + "\n"
        )
        with patch("taxotreeset.taxonomy.subprocess.run", return_value=mock_result):
            result = _resolve_name_via_ncbi("Viruses")
        assert result == "10239"
