from src.retriever import Retriever


def test_retriever_marks_define_this_as_ambiguous():
    assert Retriever._is_ambiguous("define this")


def test_retriever_accepts_specific_topic_query():
    assert not Retriever._is_ambiguous("define data heterogeneity")


def test_retriever_builds_query_variants_from_topic():
    variants = Retriever._query_variants("Define Data Heterogeneity")

    assert "Define Data Heterogeneity" in variants
    assert "data heterogeneity" in variants
    assert "what is data heterogeneity" in variants


def test_retriever_detects_heading_match():
    assert Retriever._looks_like_heading_match(
        "Define Data Heterogeneity",
        "1.3. Data Heterogeneity: Devices frequently generate and collect data.",
    )
