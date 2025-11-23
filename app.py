import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.set_page_config(page_title="Borrow vs Sell SBLOC Analyzer", layout="wide")


def run_model(
    P0: float,
    annual_spend: float,
    years: int,
    r_pct: float,
    i_pct: float,
    ltcg_rate_pct: float,
    gain_fraction: float,
    interest_pay_fraction: float,
):
    """
    Model two strategies for funding spending from a taxable portfolio:

    Strategy A (Sell):
        - Each year, sell enough to net `annual_spend` after tax.
        - Pay capital gains tax based on `ltcg_rate_pct` and `gain_fraction`.
        - Portfolio value after sale compounds at `r_pct`.

    Strategy B (Borrow):
        - Portfolio compounds at `r_pct` with no sales.
        - Each year, borrow `annual_spend` via SBLOC.
        - SBLOC interest is loan_balance * `i_pct`.
        - Fraction `interest_pay_fraction` of interest is paid from outside income;
          the remainder is capitalized into the loan.
        - Net to heirs at the end is portfolio minus loan.

    Assumes:
        - Step up in basis at death for remaining taxable portfolio.
        - No estate or inheritance tax.
        - SBLOC interest not deductible.
        - Gain fraction is constant (approximation).
    """
    r = r_pct / 100.0          # portfolio return
    i = i_pct / 100.0          # SBLOC interest rate
    ltcg_rate = ltcg_rate_pct / 100.0

    years_array = np.arange(0, years + 1, dtype=int)

    # Arrays for sell strategy
    P_sell = np.zeros(years + 1)
    tax_paid = np.zeros(years + 1)
    actual_spend_sell = np.zeros(years + 1)

    # Arrays for borrow strategy
    P_borrow = np.zeros(years + 1)
    loan = np.zeros(years + 1)
    net_borrow = np.zeros(years + 1)
    interest_paid_cash = np.zeros(years + 1)

    # Initial conditions
    P_sell[0] = P0
    P_borrow[0] = P0
    loan[0] = 0.0
    net_borrow[0] = P_borrow[0] - loan[0]
    actual_spend_sell[0] = 0.0
    interest_paid_cash[0] = 0.0

    # Effective tax per dollar sold
    eff_tax_per_dollar = ltcg_rate * gain_fraction
    eff_keep_per_dollar = 1.0 - eff_tax_per_dollar

    if eff_keep_per_dollar <= 0:
        raise ValueError(
            "Effective tax per dollar sold is 100 percent or more; "
            "check LTCG rate and gain fraction inputs."
        )

    for t in range(1, years + 1):
        # Strategy A: Sell to fund spending (annual_spend is target net consumption)
        gross_sale = annual_spend / eff_keep_per_dollar
        tax_this_year = gross_sale * eff_tax_per_dollar

        if gross_sale > P_sell[t - 1] and P_sell[t - 1] > 0:
            # Portfolio cannot support target spend; sell what remains
            gross_sale = P_sell[t - 1]
            tax_this_year = gross_sale * eff_tax_per_dollar
            net_proceeds = gross_sale - tax_this_year
        elif P_sell[t - 1] <= 0:
            # Portfolio already depleted
            gross_sale = 0.0
            tax_this_year = 0.0
            net_proceeds = 0.0
        else:
            net_proceeds = annual_spend

        P_sell[t] = max(0.0, (P_sell[t - 1] - gross_sale) * (1.0 + r))
        tax_paid[t] = tax_paid[t - 1] + tax_this_year
        actual_spend_sell[t] = net_proceeds

        # Strategy B: Borrow via SBLOC
        P_borrow[t] = P_borrow[t - 1] * (1.0 + r)

        interest = loan[t - 1] * i
        interest_paid_from_cash = interest * interest_pay_fraction
        interest_capitalized = interest * (1.0 - interest_pay_fraction)

        interest_paid_cash[t] = interest_paid_cash[t - 1] + interest_paid_from_cash

        loan[t] = loan[t - 1] + interest_capitalized + annual_spend
        net_borrow[t] = P_borrow[t] - loan[t]

    # Loan-to-value for borrow strategy
    with np.errstate(divide="ignore", invalid="ignore"):
        ltv = np.where(P_borrow > 0, loan / P_borrow, np.nan)

    df = pd.DataFrame(
        {
            "Year": years_array,
            "Sell_Portfolio": P_sell,
            "Sell_Cumulative_Tax": tax_paid,
            "Sell_Actual_Spend": actual_spend_sell,
            "Borrow_Portfolio": P_borrow,
            "Borrow_Loan_Balance": loan,
            "Borrow_Net_to_Heirs": net_borrow,
            "Borrow_LTV": ltv,
            "Borrow_Interest_Paid_From_Cash": interest_paid_cash,
        }
    )
    return df


# Streamlit UI
st.title("Borrow vs Sell SBLOC Strategy Analyzer")

st.markdown(
    """
This tool models two strategies for funding spending from a taxable investment account:

1. **Sell and pay capital gains tax each year.**  
2. **Borrow against the portfolio via an SBLOC, never sell,** then repay the loan from the portfolio at death.

By default it assumes a fixed horizon and a step up in basis at death.
"""
)

with st.sidebar:
    st.header("Inputs")

    P0 = st.number_input(
        "Initial taxable portfolio ($)",
        min_value=0.0,
        value=1_000_000.0,
        step=50_000.0,
        format="%.0f",
    )

    years = st.number_input(
        "Years until death",
        min_value=1,
        max_value=60,
        value=10,
        step=1,
    )

    spend_mode = st.radio(
        "Annual spending input",
        options=["Dollar amount", "% of starting portfolio"],
        index=0,
    )

    if spend_mode == "Dollar amount":
        annual_spend = st.number_input(
            "Target annual consumption ($, after tax)",
            min_value=0.0,
            value=40_000.0,
            step=5_000.0,
            format="%.0f",
        )
        implied_pct = (annual_spend / P0 * 100.0) if P0 > 0 else 0.0
        st.caption(f"Implied spending rate: {implied_pct:.2f} percent of initial portfolio")
    else:
        spend_pct = st.number_input(
            "Target annual consumption (% of starting portfolio, after tax)",
            min_value=0.0,
            max_value=20.0,
            value=4.0,
            step=0.5,
        )
        annual_spend = P0 * spend_pct / 100.0
        st.caption(f"Annual spending amount: ${annual_spend:,.0f}")

    r_pct = st.number_input(
        "Expected portfolio return (% per year)",
        min_value=-10.0,
        max_value=20.0,
        value=6.0,
        step=0.5,
    )

    i_pct = st.number_input(
        "SBLOC interest rate (% per year)",
        min_value=0.0,
        max_value=20.0,
        value=7.4,
        step=0.1,
    )

    ltcg_rate_pct = st.number_input(
        "Effective long term capital gains tax rate (% of gain)",
        min_value=0.0,
        max_value=50.0,
        value=23.8,
        step=0.1,
        help="Include federal LTCG, NIIT, and state if you want.",
    )

    gain_fraction = st.number_input(
        "Fraction of each sale that is gain (0 to 1)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="If basis is 40 percent of portfolio value, gain fraction is 60 percent.",
    )

    interest_pay_fraction = st.slider(
        "Fraction of SBLOC interest paid from outside income",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help=(
            "0 means all interest is capitalized into the loan. "
            "1 means you pay all interest from cash and only borrow the spending amount."
        ),
    )

    run_button = st.button("Run model", type="primary")


if run_button:
    try:
        df = run_model(
            P0=P0,
            annual_spend=annual_spend,
            years=years,
            r_pct=r_pct,
            i_pct=i_pct,
            ltcg_rate_pct=ltcg_rate_pct,
            gain_fraction=gain_fraction,
            interest_pay_fraction=interest_pay_fraction,
        )
    except ValueError as e:
        st.error(str(e))
    else:
        final = df.iloc[-1]

        max_ltv = np.nanmax(df["Borrow_LTV"]) if len(df) > 0 else np.nan
        if max_ltv > 0.5:
            st.warning(
                f"Maximum loan-to-value (LTV) reaches {max_ltv:.1%}. "
                "In practice this may create margin call risk."
            )

        tab_summary, tab_charts, tab_table = st.tabs(
            ["Summary", "Charts", "Table"]
        )

        with tab_summary:
            st.subheader("Summary at end of horizon")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Sell strategy: portfolio to heirs",
                    f"${final['Sell_Portfolio']:,.0f}",
                )
            with col2:
                st.metric(
                    "Borrow strategy: net to heirs",
                    f"${final['Borrow_Net_to_Heirs']:,.0f}",
                )
            with col3:
                diff = final["Borrow_Net_to_Heirs"] - final["Sell_Portfolio"]
                label = "Borrow minus sell"
                st.metric(label, f"${diff:,.0f}")

            st.caption(
                "Sell strategy assumes remaining portfolio receives a step up in basis at death. "
                "Borrow strategy assumes the loan is repaid from the portfolio at death and the remainder passes with a step up."
            )

            st.markdown("### Funding success (sell strategy)")
            spent_total = df["Sell_Actual_Spend"].sum()
            target_total = annual_spend * years
            st.write(
                f"Target lifetime consumption: ${target_total:,.0f} "
                f"(over {years} years at ${annual_spend:,.0f} per year)."
            )
            st.write(
                f"Actual lifetime consumption delivered by sell strategy: ${spent_total:,.0f}."
            )
            if spent_total + 1e-6 < target_total:
                st.warning(
                    "The sell strategy could not fully fund your target spending before the portfolio depleted."
                )

        with tab_charts:
            st.subheader("Portfolio and loan paths")

            fig1, ax1 = plt.subplots()
            ax1.plot(df["Year"], df["Sell_Portfolio"], label="Sell: portfolio")
            ax1.plot(df["Year"], df["Borrow_Portfolio"], label="Borrow: portfolio")
            ax1.plot(df["Year"], df["Borrow_Loan_Balance"], label="Borrow: loan balance")
            ax1.set_xlabel("Year")
            ax1.set_ylabel("Dollars")
            ax1.set_title("Portfolio and loan over time")
            ax1.yaxis.set_major_formatter(
                mtick.StrMethodFormatter('${x:,.0f}')
            )
            ax1.legend()
            st.pyplot(fig1)

            st.subheader("Net to heirs over time")

            fig2, ax2 = plt.subplots()
            ax2.plot(df["Year"], df["Sell_Portfolio"], label="Sell: portfolio to heirs")
            ax2.plot(
                df["Year"],
                df["Borrow_Net_to_Heirs"],
                label="Borrow: net to heirs (portfolio minus loan)",
            )
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Dollars")
            ax2.set_title("Net to heirs over time")
            ax2.yaxis.set_major_formatter(
                mtick.StrMethodFormatter('${x:,.0f}')
            )
            ax2.legend()
            st.pyplot(fig2)

            st.subheader("Loan-to-value (LTV) over time")

            fig3, ax3 = plt.subplots()
            ax3.plot(df["Year"], df["Borrow_LTV"], label="Borrow: LTV")
            ax3.set_xlabel("Year")
            ax3.set_ylabel("LTV")
            ax3.set_title("Loan-to-value ratio over time")
            ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0%}"))
            ax3.axhline(0.5, linestyle="--", linewidth=1)
            ax3.legend()
            st.pyplot(fig3)

        with tab_table:
            st.subheader("Year by year details")
            display_df = df.copy()
            st.dataframe(
                display_df.style.format(
                    {
                        "Sell_Portfolio": "${:,.0f}",
                        "Sell_Cumulative_Tax": "${:,.0f}",
                        "Sell_Actual_Spend": "${:,.0f}",
                        "Borrow_Portfolio": "${:,.0f}",
                        "Borrow_Loan_Balance": "${:,.0f}",
                        "Borrow_Net_to_Heirs": "${:,.0f}",
                        "Borrow_LTV": "{:.2%}",
                        "Borrow_Interest_Paid_From_Cash": "${:,.0f}",
                    }
                ),
                use_container_width=True,
            )

        st.markdown(
            """
**Important notes**

- This is a deterministic model. It does not simulate volatility, margin calls, or changing interest rates.
- It does not model estate or inheritance tax.
- It assumes step up in basis rules remain in force for the entire period.
- It treats SBLOC interest as a non deductible personal expense.
- The gain fraction is held constant as an approximation; in reality it will change as you sell.
Use it as a planning and intuition tool, not as tax or investment advice.
"""
        )
else:
    st.info("Set your assumptions in the sidebar, then click “Run model”.")
